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

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

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
            torch_dtype=config.dtype,
        )
        self.processor = AutoProcessor.from_pretrained(config.qwen_model_name)
        self.processor.tokenizer.padding_side = config.tokenizer_padding_side
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

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

        # Qwen3-VL-2B ships 267 spare embedding rows, so the `chunk_size * 4 + 1` added tokens fit
        # without a resize up to chunk_size=66. Past that, resizing changes the `embed_tokens` /
        # `lm_head` shapes and checkpoints stop loading across chunk sizes unless those prefixes are
        # in `reinit_modules` — warn instead of failing silently.
        current_rows = self.model.get_input_embeddings().weight.size(0)
        if current_rows < len(tokenizer):
            logging.warning(
                f"chunk_size={self.config.chunk_size} needs {max_action_tokens + 1} added tokens, "
                f"which exceeds the {current_rows} embedding rows of {self.config.qwen_model_name}. "
                f"Resizing to {len(tokenizer)} rows changes the shapes of "
                f"`model.qwen.model.model.language_model.embed_tokens` and "
                f"`model.qwen.model.lm_head`, so this model will not load from a checkpoint trained "
                f"with a different chunk_size unless those prefixes are in `reinit_modules`."
            )
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
        # Images are on-device float [0, 1] tensors from `to_pixel_values` (#3750), not PIL — the
        # whole adapter layer (`VLAJEPAPolicy._prepare_model_inputs`) feeds this pipeline.
        images: Sequence[Sequence[torch.Tensor]],
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
            # On-device, do_rescale=False: images already arrive as float in [0, 1] via
            # `to_pixel_values` (#3750), so skip the processor's /255 rescale and GPU roundtrip.
            batch_inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                processor_kwargs={
                    "padding": True,
                    "return_tensors": "pt",
                    "device": self.model.device,
                    "do_rescale": False,
                },
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
            do_rescale=False,
            device=self.model.device,
            padding=True,
            return_tensors="pt",
        )
        return batch_inputs.to(self.model.device)

    @staticmethod
    def to_pixel_values(image_tensor: torch.Tensor) -> torch.Tensor:
        """Prepare an image/video tensor for the fast processors (used with do_rescale=False).

        The dataset decoder yields float32 in [0, 1] (channels-first) and VISUAL
        normalization is IDENTITY, so the tensor already arrives in [0, 1]; we pass it
        through as float and let the processors normalize (no rescale, no uint8
        quantization). A single channel is expanded to 3 to match the RGB processors.

        Works for any channels-first layout (channel dim is -3): [C, H, W], [B, C, H, W],
        [T, C, H, W], [B, V, T, C, H, W], ...
        """
        image = image_tensor.detach().float()
        if image.shape[-3] == 1:
            repeats = [1] * image.ndim
            repeats[-3] = 3
            image = image.repeat(*repeats)
        return image
