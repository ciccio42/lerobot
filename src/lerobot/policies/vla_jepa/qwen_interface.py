# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
else:
    AutoProcessor = None
    Qwen3VLForConditionalGeneration = None

from .configuration_vla_jepa import VLAJEPAConfig


class Qwen3VLInterface(torch.nn.Module):
    def __init__(self, config: VLAJEPAConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.qwen_model_name,
            torch_dtype=self._get_torch_dtype(config.torch_dtype),
        )
        self.processor = AutoProcessor.from_pretrained(config.qwen_model_name)
        self.processor.tokenizer.padding_side = config.tokenizer_padding_side
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

    @staticmethod
    def _get_torch_dtype(dtype_name: str) -> torch.dtype:
        if dtype_name == "float32":
            return torch.float32
        if dtype_name == "float16":
            return torch.float16
        return torch.bfloat16

    def expand_tokenizer(self) -> tuple[list[str], list[int], int]:
        # starVLA/JEVLA checkpoints expand action tokens as action_horizon * 4,
        # independent of vj2 num_action_tokens_per_timestep. Keeping this count
        # is required for Qwen embedding/lm_head checkpoint shapes to match.
        max_action_tokens = self.config.chunk_size * 4
        tokenizer = self.processor.tokenizer
        action_tokens = []
        action_token_ids = []
        for idx in range(max_action_tokens):
            token = self.config.special_action_token.format(idx)
            action_tokens.append(token)
            if token not in tokenizer.get_vocab():
                tokenizer.add_tokens([token], special_tokens=True)
            action_token_ids.append(tokenizer.convert_tokens_to_ids(token))

        embodied_action_token = self.config.embodied_action_token
        if embodied_action_token not in tokenizer.get_vocab():
            tokenizer.add_tokens([embodied_action_token], special_tokens=True)
        embodied_action_token_id = tokenizer.convert_tokens_to_ids(embodied_action_token)

        # Added last so the ids above (and the checkpoint's embedding rows) are unchanged.
        if self.config.use_demo and self.config.demo_encoder == "vjepa":
            if self.config.demo_token not in tokenizer.get_vocab():
                tokenizer.add_tokens([self.config.demo_token], special_tokens=True)

        if self.model.get_input_embeddings().weight.size(0) < len(tokenizer):
            self.model.resize_token_embeddings(len(tokenizer))
        return action_tokens, action_token_ids, embodied_action_token_id

    def _prompt(
        self, instruction: str | None, action_prompt: str, embodied_prompt: str, demo_prompt: str
    ) -> str:
        if not self.config.use_demo:
            return self.config.prompt_template.format(
                instruction=instruction, actions=action_prompt, e_actions=embodied_prompt
            )
        instruction_part = (
            self.config.demo_instruction_template.format(instruction=instruction)
            if self.config.conditioning == "both" and instruction is not None
            else ""
        )
        return self.config.demo_prompt_template.format(
            demo=demo_prompt, instruction=instruction_part, actions=action_prompt, e_actions=embodied_prompt
        )

    def build_inputs(
        self,
        images: Sequence[Sequence[Image.Image]],
        instructions: Sequence[str | None],
        action_prompt: str,
        embodied_prompt: str,
        demo_prompt: str = "",
        demo_videos: Sequence[tuple[np.ndarray, float]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build Qwen3-VL inputs.

        demo_prompt: text put at the `{demo}` slot of `demo_prompt_template` (the <|demo|> placeholders
            for the V-JEPA demo encoder).
        demo_videos: per-sample (frames [T, H, W, 3] uint8, fps) for the native Qwen video demo encoder;
            the video is inserted at the `{demo}` slot.
        """
        video_marker = "\x00DEMO_VIDEO\x00"
        messages = []
        for sample_images, instruction in zip(images, instructions, strict=True):
            prompt = self._prompt(
                instruction,
                action_prompt,
                embodied_prompt,
                video_marker if demo_videos is not None else demo_prompt,
            )
            content = [{"type": "image", "image": img} for img in sample_images]
            if demo_videos is not None:
                before, after = prompt.split(video_marker)
                content += [
                    {"type": "text", "text": before},
                    {"type": "video"},
                    {"type": "text", "text": after},
                ]
            else:
                content.append({"type": "text", "text": prompt})
            messages.append([{"role": "user", "content": content}])

        if demo_videos is None:
            batch_inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                processor_kwargs={"padding": True, "return_tensors": "pt"},
            )
            return batch_inputs.to(self.model.device)

        # Pre-sampled demo frames: pass them to the processor directly, with metadata so that the
        # Qwen3-VL frame timestamps in the prompt reflect the demo's real timing.
        texts = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch_inputs = self.processor(
            text=texts,
            images=[img for sample_images in images for img in sample_images],
            videos=[frames for frames, _ in demo_videos],
            videos_kwargs={
                "do_sample_frames": False,
                "video_metadata": [
                    {"fps": fps, "total_num_frames": len(frames), "frames_indices": list(range(len(frames)))}
                    for frames, fps in demo_videos
                ],
            },
            padding=True,
            return_tensors="pt",
        )
        return batch_inputs.to(self.model.device)

    @staticmethod
    def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
        image = image_tensor.detach().cpu()
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = image.permute(1, 2, 0)
        image = image.float()
        if image.max() <= 1.0:
            image = image * 255.0
        image = image.clamp(0, 255).round().to(torch.uint8).numpy()
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        return Image.fromarray(image)
