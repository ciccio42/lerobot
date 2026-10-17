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

import json
import logging
import os
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from safetensors.torch import load_file
from torch import Tensor, nn

from lerobot.configs import FeatureType
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.policies.utils import log_model_loading_keys, populate_queues
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.device_utils import get_autocast_context, resolve_safetensors_device
from lerobot.utils.import_utils import _transformers_available, require_package

from ..chunk import ChunkPolicySpec, FeatureSpec

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoModel, AutoVideoProcessor
else:
    AutoModel = None
    AutoVideoProcessor = None

from .action_head import VLAJEPAActionHead
from .configuration_vla_jepa import VLAJEPAConfig
from .demo_bank import DemoVideoBank, normalize_instruction
from .demo_resampler import DemoResampler
from .qwen_interface import Qwen3VLInterface
from .world_model import ActionConditionedVideoPredictor

# ============================================================================
# Native VLA-JEPA Model - follows original starVLA VLA_JEPA.py implementation
# ============================================================================


class VLAJEPAModel(nn.Module):
    """
    Native VLA-JEPA model following the original starVLA VLA_JEPA.py.

    Components:
      - Qwen3-VL: vision-language backbone for fused embeddings
      - DiT-B: flow-matching action head for future action prediction
      - V-JEPA: world model for video frame prediction

    Inputs are batched tensors kept on the model device
      - images: List[List[Tensor [C, H, W]]] (float [0,1]) — per sample, per view (Qwen messages)
      - instructions: List[str]
      - videos: Tensor [B, V, T, C, H, W] (float [0,1], world model only)
      - actions: Tensor [B, T, action_dim] (optional, training only)
      - state: Tensor [B, 1, state_dim] (optional)
      - action_is_pad: Tensor [B, T] (optional)
    """

    def __init__(self, config: VLAJEPAConfig) -> None:
        super().__init__()
        require_package("transformers", extra="vla_jepa")
        self.config = config

        # Vision-language backbone
        self.qwen = Qwen3VLInterface(config)

        # Tokenizer expansion for special action tokens
        self.action_tokens, self.action_token_ids, self.embodied_action_token_id = (
            self.qwen.expand_tokenizer()
        )
        self.register_buffer(
            "_action_token_ids_t",
            torch.tensor(self.action_token_ids, dtype=torch.long),
            persistent=False,
        )

        # Action head (flow-matching DiT)
        self.action_model = VLAJEPAActionHead(config, cross_attention_dim=self.qwen.model.config.hidden_size)

        # JEPA world model components (the V-JEPA encoder is also the demo encoder in option A)
        use_vjepa_demo = config.use_demo and config.demo_encoder == "vjepa"
        if config.enable_world_model or use_vjepa_demo:
            self.video_encoder = AutoModel.from_pretrained(
                config.jepa_encoder_name,
                torch_dtype=config.dtype,
            )
            self.video_processor = AutoVideoProcessor.from_pretrained(config.jepa_encoder_name)
        else:
            self.video_encoder = None
            self.video_processor = None

        if config.enable_world_model:
            num_views = config.num_world_model_views
            tubelet_size = self.video_encoder.config.tubelet_size
            image_size = getattr(self.video_encoder.config, "image_size", None)
            if image_size is None:
                first_image_shape = next(iter(config.image_features.values())).shape
                image_size = first_image_shape[-1]
            self.video_predictor: ActionConditionedVideoPredictor | None = ActionConditionedVideoPredictor(
                num_frames=config.num_video_frames // tubelet_size,
                img_size=(image_size, image_size),
                patch_size=16,
                tubelet_size=1,
                embed_dim=self.video_encoder.config.hidden_size * num_views,
                action_embed_dim=self.qwen.model.config.hidden_size,
                predictor_embed_dim=self.video_encoder.config.hidden_size,
                depth=config.predictor_depth,
                num_heads=config.predictor_num_heads,
                mlp_ratio=config.predictor_mlp_ratio,
                num_action_tokens_per_step=config.num_action_tokens_per_timestep,
                dropout=config.predictor_dropout,
            )
        else:
            self.video_predictor = None

        # Demo-video conditioning (option A): V-JEPA2 demo features -> K <|demo|> token embeddings
        if use_vjepa_demo:
            self.demo_token_id = self.qwen.processor.tokenizer.convert_tokens_to_ids(config.demo_token)
            embed_weight = self.qwen.model.get_input_embeddings().weight
            self.demo_resampler = DemoResampler(
                in_dim=self.video_encoder.config.hidden_size,
                out_dim=self.qwen.model.config.hidden_size,
                num_tokens=config.demo_num_tokens,
                depth=config.demo_resampler_depth,
                num_heads=config.demo_resampler_num_heads,
                embed_rms=embed_weight.detach().float().pow(2).mean().sqrt().item(),
            )
        else:
            self.demo_token_id = None
            self.demo_resampler = None

        # Demo-text contrastive loss: projection head on the mean demo token, learned temperature.
        # The text anchors are built by build_demo_text_anchors() (training only).
        if use_vjepa_demo and config.demo_contrastive_weight > 0:
            hidden = self.qwen.model.config.hidden_size
            # Output matches the text anchor: [mean-pooled part, last-token part], 2 * hidden.
            self.demo_proj = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2 * hidden))
            self.demo_logit_scale = nn.Parameter(torch.tensor(np.log(1.0 / config.demo_contrastive_temperature)))
        else:
            self.demo_proj = None
            self.demo_logit_scale = None
        self.register_buffer("demo_text_anchors", torch.empty(0), persistent=False)
        self.register_buffer("demo_anchor_task_ids", torch.empty(0, dtype=torch.long), persistent=False)

        if config.freeze_qwen:
            self.qwen.requires_grad_(False)

        # Build prompt placeholders.
        # Use the encoder's actual tubelet_size when available (world model enabled),
        # otherwise fall back to config.
        _tubelet_size = (
            self.video_encoder.config.tubelet_size
            if config.enable_world_model
            else self.config.jepa_tubelet_size
        )
        num_action_prompt_steps = self.config.num_video_frames // _tubelet_size - 1
        self.replace_prompt = "".join(
            token * self.config.num_action_tokens_per_timestep
            for token in self.action_tokens[:num_action_prompt_steps]
        )
        self.embodied_replace_prompt = (
            self.config.embodied_action_token * self.config.num_embodied_action_tokens_per_instruction
        )

    def _qwen_last_decoder_hidden(
        self, qwen_inputs: dict[str, torch.Tensor], demo_embeds: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the last decoder hidden state before the final RMSNorm.

        The model was trained on the last block's pre-RMSNorm output, but in transformers 5.x
        `hidden_states[-1]` is post-norm, so hook `language_model.layers[-1]` instead.

        Calls the inner `Qwen3VLModel`, not the `Qwen3VLForConditionalGeneration` wrapper, whose
        forward would build and discard full-sequence logits over the 151936-token vocab (~3.4 GB
        in bf16 at batch 8). The wrapper stays as `self.qwen.model` so `lm_head` keeps its
        checkpoint key; only this path skips it.
        """
        captured: list[torch.Tensor] = []

        def _hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            captured.append(h)

        def _inject_demo(module, inputs, output):
            # Replace the embeddings of the <|demo|> placeholders with the resampled demo tokens.
            mask = inputs[0] == self.demo_token_id
            output = output.clone()
            output[mask] = demo_embeds.reshape(-1, output.shape[-1]).to(output.dtype)
            return output

        last_layer = self.qwen.model.model.language_model.layers[-1]
        handles = [last_layer.register_forward_hook(_hook)]
        if demo_embeds is not None:
            handles.append(self.qwen.model.get_input_embeddings().register_forward_hook(_inject_demo))
        try:
            self.qwen.model.model(**qwen_inputs)
        finally:
            for handle in handles:
                handle.remove()

        return captured[0]  # [B, seq_len, H]

    @torch.no_grad()
    def build_demo_text_anchors(self, instructions: list[str], task_ids: list[int]) -> None:
        """Text anchors t_c from the frozen base Qwen, one per training task instruction.

        t_c = [mean hidden state at layer `demo_anchor_mean_layer`, last-token hidden state at layer
        `demo_anchor_last_layer`], each centred over the tasks and L2-normalised (see config).
        Called once at construction, before any checkpoint weights are loaded, so the anchors always come
        from the same pretrained Qwen (identical on every rank and after every resume) and never drift.
        """
        tokenizer = self.qwen.processor.tokenizer  # left padding: position -1 is the last real token
        enc = tokenizer(list(instructions), padding=True, return_tensors="pt")
        language_model = self.qwen.model.model.language_model
        device = next(language_model.parameters()).device
        out = language_model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
            output_hidden_states=True,
        )
        mask = enc["attention_mask"].to(device).unsqueeze(-1).float()
        mean_part = (out.hidden_states[self.config.demo_anchor_mean_layer].float() * mask).sum(1) / mask.sum(1)
        last_part = out.hidden_states[self.config.demo_anchor_last_layer][:, -1].float()

        def centre_norm(x: Tensor) -> Tensor:
            return F.normalize(x - x.mean(0, keepdim=True), dim=-1)

        self.demo_text_anchors = torch.cat([centre_norm(mean_part), centre_norm(last_part)], dim=-1).cpu()
        self.demo_anchor_task_ids = torch.tensor(task_ids, dtype=torch.long)

    def _demo_text_contrastive(self, demo_embeds: Tensor, labels: Tensor) -> dict[str, Tensor]:
        """InfoNCE between the projected mean demo token and all text anchors, plus similarity metrics."""
        valid = labels >= 0
        z = F.normalize(self.demo_proj(demo_embeds.float().mean(1)), dim=-1)  # [B, H]
        t = F.normalize(self.demo_text_anchors.to(z.device).float(), dim=-1)  # [C, H]
        sims = z @ t.T  # [B, C] cosine similarities
        logits = sims * self.demo_logit_scale.exp().clamp(max=100.0)
        loss = F.cross_entropy(logits, labels.clamp(min=0), reduction="none")
        loss = (loss * valid).sum() / valid.sum().clamp(min=1)

        out = {"demo_contrastive_loss": loss * self.config.demo_contrastive_weight}
        with torch.no_grad():
            pred = sims.argmax(-1)
            lab = labels.clamp(min=0)
            pos = sims.gather(1, lab[:, None]).squeeze(1)
            neg = sims.masked_fill(F.one_hot(lab, sims.shape[1]).bool(), -2.0).max(-1).values
            n = valid.sum().clamp(min=1)
            out["demo_text_acc"] = ((pred == lab) & valid).sum() / n
            out["demo_text_margin"] = ((pos - neg) * valid).sum() / n
            out["demo_text_pos_sim"] = (pos * valid).sum() / n
            out["demo_text_temperature"] = 1.0 / self.demo_logit_scale.exp().clamp(max=100.0)
            if self.config.demo_task_factors and self.demo_anchor_task_ids.numel():
                ids = self.demo_anchor_task_ids.to(lab.device)
                pred_id, true_id = ids[pred], ids[lab]
                for name, size in reversed(list(self.config.demo_task_factors.items())):
                    out[f"demo_text_{name}_acc"] = (((pred_id % size) == (true_id % size)) & valid).sum() / n
                    pred_id, true_id = pred_id // size, true_id // size
        return out

    def _encode_demos(
        self, demos: list[tuple[np.ndarray, float]] | None
    ) -> tuple[str, list[tuple[np.ndarray, float]] | None, Tensor | None]:
        """Return (demo_prompt, demo_videos for Qwen, demo token embeddings) for build_inputs."""
        if not self.config.use_demo:
            return "", None, None
        if demos is None or any(d is None for d in demos):
            raise ValueError("conditioning uses a demo video but a sample has no 'demo'.")
        if self.config.demo_encoder == "qwen_video":
            return "", demos, None

        frames = np.stack([d[0] for d in demos]).transpose(0, 1, 4, 2, 3)  # [B, T, 3, S, S]
        pixels = self.video_processor(videos=list(frames), return_tensors="pt")["pixel_values_videos"].to(
            self.video_encoder.device
        )
        with torch.no_grad():
            feats = self.video_encoder.get_vision_features(pixel_values_videos=pixels)  # [B, T'*P, D]
        num_temporal = frames.shape[1] // self.video_encoder.config.tubelet_size
        demo_embeds = self.demo_resampler(feats.float(), num_temporal)  # [B, K, H]
        return self.config.demo_token * self.config.demo_num_tokens, None, demo_embeds

    # ---- Native VLA-JEPA forward (follows original VLA_JEPA.py) ----

    def _encode_qwen(
        self,
        images: list[list[Tensor]],
        instructions: list[str | None],
        demos: list[tuple[np.ndarray, float]] | None = None,
        *,
        need_action_tokens: bool,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """Run Qwen and gather the embodied-action (and optionally action) token hidden states.

        Returns (embodied_action_tokens, action_tokens, demo_embeds) — `demo_embeds` is the
        resampled V-JEPA demo-token embedding (for the demo-text contrastive loss), or None when
        `demos` is None / conditioning doesn't use a demo.
        """
        demo_prompt, demo_videos, demo_embeds = self._encode_demos(demos)
        qwen_inputs = self.qwen.build_inputs(
            images=images,
            instructions=instructions,
            action_prompt=self.replace_prompt,
            embodied_prompt=self.embodied_replace_prompt,
            demo_prompt=demo_prompt,
            demo_videos=demo_videos,
        )
        input_ids = qwen_inputs["input_ids"]
        embodied_idx = (input_ids == self.embodied_action_token_id).nonzero(as_tuple=True)
        action_idx = None
        if need_action_tokens:
            action_mask = torch.isin(input_ids, self._action_token_ids_t)
            action_idx = action_mask.nonzero(as_tuple=True)

        device_type = next(self.parameters()).device.type
        with get_autocast_context(device_type, torch.bfloat16):
            last_hidden = self._qwen_last_decoder_hidden(qwen_inputs, demo_embeds)  # [B, seq_len, H]
            b, _, h = last_hidden.shape
            embodied_action_tokens = last_hidden[embodied_idx[0], embodied_idx[1], :].view(b, -1, h)
            action_tokens = (
                last_hidden[action_idx[0], action_idx[1], :].view(b, -1, h)
                if action_idx is not None
                else None
            )
        return embodied_action_tokens, action_tokens, demo_embeds

    def _causal_video_embeddings(self, video_pixels: Tensor, tubelet_size: int, num_positions: int) -> Tensor:
        """Encode `num_positions` leading temporal positions from only their own raw-frame prefix.

        A single V-JEPA2 pass over the full clip lets bidirectional attention leak future frames
        into every position's embedding, including the context positions used as predictor input
        (#4153). Running one prefix-only pass per context position keeps position i blind to frames
        after it, at the cost of `num_positions` encoder calls instead of one.
        """
        positions = []
        for position in range(num_positions):
            prefix = self.video_encoder.get_vision_features(
                pixel_values_videos=video_pixels[:, : (position + 1) * tubelet_size]
            )
            tokens_per_position = prefix.shape[1] // (position + 1)
            positions.append(prefix[:, -tokens_per_position:])
        return torch.cat(positions, dim=1)

    @staticmethod
    def _merge_views(embeddings: Tensor, b: int, v: int) -> Tensor:
        """Merge per-view features: [B*V, N, H] -> [B, N, V*H].

        Rows run view-fastest, since `videos.reshape(b * v, ...)` flattens (B, V) row-major. A
        `chunk(chunks=v, dim=0)` + `cat(dim=2)` merge assumes view-slowest and so concatenates
        features from *different* samples — shape-valid, so it fails silently for b > 1.
        """
        n_tokens, hidden = embeddings.shape[1], embeddings.shape[2]
        return embeddings.reshape(b, v, n_tokens, hidden).permute(0, 2, 1, 3).reshape(b, n_tokens, v * hidden)

    def _world_model_loss(self, videos: Tensor, action_tokens: Tensor, reduction: str = "mean") -> Tensor:
        """JEPA encode + predictor L1 loss. `videos` is [B, V, T, C, H, W] float in [0, 1].

        `reduction="none"` returns a per-sample loss (B,) for sample weighting (RA-BC);
        "mean" returns the scalar loss.
        """
        if self.video_predictor is None:
            raise RuntimeError("The world model loss requires `enable_world_model=True`.")
        # Match the world model's expected view count: pad with the first view, or trim extras.
        num_views = self.config.num_world_model_views
        if videos.shape[1] < num_views:
            missing = num_views - videos.shape[1]
            videos = torch.cat([videos, videos[:, :1].repeat(1, missing, 1, 1, 1, 1)], dim=1)
        elif videos.shape[1] > num_views:
            videos = videos[:, :num_views]

        b, v, t_frames, c, h_img, w_img = videos.shape
        flat = videos.reshape(b * v, t_frames, c, h_img, w_img)
        # Fast (torchvision) video processor on-device, do_rescale=False (frames already in [0, 1]).
        video_pixels = self.video_processor(
            videos=list(flat),
            return_tensors="pt",
            device=self.video_encoder.device,
            do_rescale=False,
        )["pixel_values_videos"]  # [B*V, T, C, H, W]

        tubelet_size = self.video_encoder.config.tubelet_size
        with torch.no_grad():
            video_embeddings = self.video_encoder.get_vision_features(pixel_values_videos=video_pixels)
            video_embeddings = self._merge_views(video_embeddings, b, v)

        # num_video_frames raw frames → t_enc_total temporal positions after tubelet compression
        t_enc_total = self.config.num_video_frames // tubelet_size
        if t_enc_total < 2:
            zero_shape = (video_embeddings.shape[0],) if reduction == "none" else ()
            return torch.zeros(zero_shape, device=video_embeddings.device)

        # Shift-by-one JEPA split: input_states = positions 0..T-2, gt_states = positions 1..T-1
        t_enc_ctx = t_enc_total - 1
        tokens_per_frame = video_embeddings.shape[1] // t_enc_total
        if self.config.causal_world_model_context:
            # The shared pass above lets bidirectional attention leak future frames into the context
            # positions used as predictor input (#4153). Recompute input_states causally instead;
            # gt_states keeps the full-pass embeddings (a target encoder seeing full context is fine).
            with torch.no_grad():
                input_states = self._causal_video_embeddings(video_pixels, tubelet_size, t_enc_ctx)
                input_states = self._merge_views(input_states, b, v)
        else:
            input_states = video_embeddings[:, : tokens_per_frame * t_enc_ctx, :]
        gt_states = video_embeddings[:, tokens_per_frame:, :]

        expected_actions = t_enc_ctx * self.config.num_action_tokens_per_timestep
        if action_tokens.shape[1] < expected_actions:
            pad = action_tokens[:, -1:].repeat(1, expected_actions - action_tokens.shape[1], 1)
            action_tokens = torch.cat([action_tokens, pad], dim=1)

        predicted_states = self.video_predictor(
            input_states.float(), action_tokens[:, :expected_actions].float()
        )
        if reduction == "none":
            # Per-sample loss (B,): mean over all non-batch dims (tokens, feature).
            elementwise = F.l1_loss(predicted_states, gt_states.float(), reduction="none")
            return elementwise.mean(dim=tuple(range(1, elementwise.ndim)))
        return F.l1_loss(predicted_states, gt_states.float(), reduction="mean")

    def _action_loss(
        self,
        embodied_action_tokens: Tensor,
        actions: Tensor,
        state: Tensor | None,
        action_is_pad: Tensor | None,
        reduction: str = "mean",
    ) -> Tensor:
        """Flow-matching action-head loss, repeated over `repeated_diffusion_steps`.

        `reduction="none"` returns a per-sample loss (B,) — the `repeated_diffusion_steps`
        independent noise draws are averaged back per original sample — for RA-BC weighting.
        """
        device_type = next(self.parameters()).device.type
        with get_autocast_context(device_type, torch.float32):
            r = self.config.repeated_diffusion_steps
            horizon = self.config.chunk_size
            b = embodied_action_tokens.shape[0]
            actions_target = actions[:, -horizon:, :].to(torch.float32).repeat(r, 1, 1)
            embodied = embodied_action_tokens.repeat(r, 1, 1)
            state_rep = state.to(embodied_action_tokens.dtype).repeat(r, 1, 1) if state is not None else None
            pad_rep = action_is_pad[:, -horizon:].repeat(r, 1) if action_is_pad is not None else None
            loss = self.action_model(embodied, actions_target, state_rep, pad_rep, reduction=reduction)
            if reduction == "none":
                # `.repeat(r, 1, 1)` tiles as [rep0(b0..b_{B-1}), rep1(...), ...] → (r, B); mean over reps.
                return loss.view(r, b).mean(dim=0)
            return loss

    def forward(
        self,
        images: list[list[Tensor]],
        instructions: list[str | None],
        videos: Tensor | None = None,
        actions: Tensor | None = None,
        state: Tensor | None = None,
        action_is_pad: Tensor | None = None,
        demos: list[tuple[np.ndarray, float]] | None = None,
        demo_labels: list[int] | None = None,
        reduction: str = "mean",
    ) -> dict[str, Tensor]:
        """Native forward: Qwen encode → optional world-model loss → optional action-head loss.

        `reduction="none"` makes both loss terms per-sample (B,) for RA-BC weighting; "mean"
        returns scalar losses. `demos`/`demo_labels` are only used when `config.use_demo`.
        """
        embodied_action_tokens, action_tokens, demo_embeds = self._encode_qwen(
            images, instructions, demos, need_action_tokens=self.config.enable_world_model
        )
        demo_outputs: dict[str, Tensor] = {}
        if self.demo_proj is not None and demo_embeds is not None and self.demo_text_anchors.numel():
            labels = torch.tensor(demo_labels, device=demo_embeds.device, dtype=torch.long)
            demo_outputs = self._demo_text_contrastive(demo_embeds, labels)

        if self.config.enable_world_model and videos is not None:
            wm_loss = self._world_model_loss(videos, action_tokens, reduction=reduction)
        else:
            zero_shape = (embodied_action_tokens.shape[0],) if reduction == "none" else ()
            wm_loss = torch.zeros(zero_shape, device=embodied_action_tokens.device)

        if actions is None:
            return {"wm_loss": wm_loss, **demo_outputs}

        action_loss = self._action_loss(
            embodied_action_tokens, actions, state, action_is_pad, reduction=reduction
        )
        return {
            "action_loss": action_loss,
            "wm_loss": wm_loss * self.config.world_model_loss_weight,
            **demo_outputs,
        }

    # ---- Native predict_action (follows original VLA_JEPA.predict_action) ----

    @torch.no_grad()
    def predict_action(
        self,
        images: list[list[Tensor]],
        instructions: list[str | None],
        state: Tensor | None = None,
        demos: list[tuple[np.ndarray, float]] | None = None,
    ) -> Tensor:
        """Predict an action chunk. `images` is per-sample, per-view float [0,1] [C, H, W] tensors.

        `demos` is required when `config.conditioning` uses a demo.
        """
        if self.config.resize_images_to is not None:
            height, width = self.config.resize_images_to
            images = [
                [F.interpolate(img[None], size=(height, width), mode="area")[0] for img in views]
                for views in images
            ]

        embodied_action_tokens, _, _ = self._encode_qwen(
            images, instructions, demos, need_action_tokens=False
        )
        return self.action_model.predict_action(
            embodied_action_tokens.float(), state.float() if state is not None else None
        )

# ============================================================================
# LeRobot Adapter Layer - converts between LeRobot batch format and native VLA-JEPA format
# ============================================================================


class VLAJEPAPolicy(PreTrainedPolicy):
    """
    LeRobot adapter for VLA-JEPA.

    Converts LeRobot's standard batch format (dict[str, Tensor]) to the batched tensors
    the native model expects (keeping everything on-device), calls the native model, and
    converts outputs back to LeRobot format.
    """

    config_class = VLAJEPAConfig
    name = "vla_jepa"

    def __init__(self, config: VLAJEPAConfig, **kwargs) -> None:
        super().__init__(config)
        config.validate_features()
        # Dataset dim derivation lives in `VLAJEPAConfig.set_dataset_feature_metadata` (called by
        # `make_policy` before this): keeps `__init__` from mutating a config it does not own, and
        # makes the derived dims visible to the processor factory too.
        self.model = VLAJEPAModel(config)

        self.demo_bank = None
        if config.use_demo:
            if config.demo_root is None or config.demo_task_map_path is None:
                raise ValueError("conditioning uses a demo: set `demo_root` and `demo_task_map_path`.")
            self.demo_bank = DemoVideoBank(
                root=config.demo_root,
                task_map_path=config.demo_task_map_path,
                num_frames=config.demo_num_frames,
                image_size=config.demo_image_size,
                num_eval_per_task=config.demo_num_eval_per_task,
                cache_videos=config.demo_cache_videos,
            )
        self._demo_rng = np.random.default_rng()

        # Text anchors for the demo-text contrastive loss: one per training task of the dataset.
        self._demo_label_index: dict[str, int] = {}
        if self.model.demo_proj is not None and (dataset_meta := kwargs.get("dataset_meta")) is not None:
            names = [str(t) for t in dataset_meta.tasks.index]
            config.demo_contrastive_task_names = names
            task_ids = [int(self.demo_bank.task_folder(n).split("_")[-1]) for n in names]
            self.model.build_demo_text_anchors(names, task_ids)
            self._demo_label_index = {normalize_instruction(n): i for i, n in enumerate(names)}
        self.reset()

    def reset(self) -> None:
        self._queues: dict[str, deque[Tensor]] = {ACTION: deque(maxlen=self.config.n_action_steps)}
        # The demo is chosen once per episode and kept for all of its steps.
        self._episode_demos: dict[str, tuple[np.ndarray, float]] = {}

    def chunk_inference_spec(self) -> ChunkPolicySpec:
        """Serve current frames; future video observations are training targets only.

        The world model stays configured as trained. ``_prepare_model_inputs``
        builds its video targets only for training, while direct prediction uses
        current images, language and state without reading the playback queue.
        """
        if self.config.n_obs_steps != 1:
            raise ValueError("VLA-JEPA chunk serving requires n_obs_steps=1.")
        return ChunkPolicySpec(
            prediction_steps=self.config.chunk_size,
            execution_steps=self.config.n_action_steps,
        )

    def validate_chunk_input_features(self, features: tuple[FeatureSpec, ...]) -> None:
        """Allow the same configured image resizing as direct local inference."""
        expected = self.config.input_features or {}
        supplied = {feature.name: feature for feature in features}
        if supplied.keys() != expected.keys():
            raise ValueError("Canonical feature names must exactly match policy input_features.")
        for name, policy_feature in expected.items():
            feature = supplied[name]
            if policy_feature.type == FeatureType.VISUAL:
                shape = policy_feature.shape
                if len(shape) != 3 or shape[0] != 3 or feature.kind != "rgb":
                    raise ValueError(f"VLA-JEPA requires RGB camera inputs: {name}.")
                if not self.config.resize_images_to and feature.shape != (shape[1], shape[2], shape[0]):
                    raise ValueError(f"RGB feature shape differs from checkpoint: {name}.")
            elif feature.kind != "tensor" or feature.shape != tuple(policy_feature.shape):
                raise ValueError(f"Tensor feature shape differs from checkpoint: {name}.")

    def _get_demo(self, instruction: str) -> tuple[np.ndarray, float]:
        if self.training:
            return self.demo_bank.sample(instruction, training=True, rng=self._demo_rng)
        if instruction not in self._episode_demos:
            self._episode_demos[instruction] = self.demo_bank.sample(instruction, training=False)
            self._save_test_demo(instruction, self._episode_demos[instruction])
        return self._episode_demos[instruction]

    def _save_test_demo(self, instruction: str, demo: tuple[np.ndarray, float]) -> None:
        """At test time only, if VLA_JEPA_DEMO_SAVE_DIR is set, save the demo frames given to the model."""
        save_root = os.environ.get("VLA_JEPA_DEMO_SAVE_DIR")
        if not save_root:
            return
        self._saved_demo_count = getattr(self, "_saved_demo_count", 0) + 1
        info = dict(self.demo_bank.last_sample or {})
        out = Path(save_root) / f"episode_{self._saved_demo_count:04d}_{info.get('task_folder', 'demo')}"
        out.mkdir(parents=True, exist_ok=True)
        frames, fps = demo
        for k, frame in enumerate(frames):
            Image.fromarray(frame).save(out / f"frame_{k:02d}.png")
        strip = np.concatenate(list(frames), axis=1)  # [S, T*S, 3]
        Image.fromarray(strip).save(out / "strip.png")
        info.update({"instruction": instruction, "fps": fps, "shape": list(frames.shape)})
        (out / "demo.json").write_text(json.dumps(info, indent=2))

    # ---- Format Conversion: LeRobot → Native ----

    def _prepare_model_inputs(self, batch: dict[str, Tensor], training=True) -> dict[str, Any]:
        """Convert a LeRobot batch to the model's batched, on-device inputs.

        LeRobot format:
            batch = {
                "observation.images.<key>": Tensor [B, C, H, W] or [B, T, C, H, W],
                "observation.state": Tensor [B, state_dim] or [B, T, state_dim],
                "action": Tensor [B, chunk_size, action_dim],  (training only)
                "task": str | List[str],  (optional instruction)
            }

        Returns the kwargs for `VLAJEPAModel.forward` / `.predict_action` (everything stays
        on the batch device; no per-sample shredding): `images` (per-sample, per-view list for
        Qwen messages), `instructions`, and the batched `videos` / `actions` / `state` /
        `action_is_pad` when present.
        """
        image_keys = list(self.config.image_features.keys())
        if not image_keys:
            raise ValueError("VLAJEPA requires at least one image feature.")
        batch_size = batch[image_keys[0]].shape[0]

        # Current-frame image per view ([B, C, H, W]); regroup per sample for Qwen messages. Resize to
        # `resize_images_to` as `predict_action` does, so training and inference feed Qwen the same
        # resolution and native frames (e.g. 720x1280) don't blow up the vision-tower patch count.
        resize_hw = tuple(self.config.resize_images_to) if self.config.resize_images_to else None
        frames = []
        for key in image_keys:
            t = batch[key]
            if t.ndim == 5:  # [B, T, C, H, W] -> current observation (delta=0)
                t = t[:, 0]
            px = self.model.qwen.to_pixel_values(t)  # [B, C, H, W]
            if resize_hw is not None and tuple(px.shape[-2:]) != resize_hw:
                px = F.interpolate(px.float(), size=resize_hw, mode="area")
            frames.append(px)
        images = [[frame[b] for frame in frames] for b in range(batch_size)]

        tasks = batch.get("task")
        if tasks is None:
            instructions = ["Execute the robot action."] * batch_size
        elif isinstance(tasks, str):
            instructions = [tasks] * batch_size
        else:
            instructions = list(tasks)

        # ---- Collect demo videos (selected by task instruction) ----
        if self.config.use_demo:
            demos = [self._get_demo(instruction) for instruction in instructions]
            # Contrastive label = index of the task's own text anchor (before any language dropout).
            demo_labels = [self._demo_label_index.get(normalize_instruction(i), -1) for i in instructions]
            if self.config.conditioning == "both" and self.training and self.config.language_dropout > 0:
                keep = self._demo_rng.random(batch_size) >= self.config.language_dropout
                instructions = [i if k else None for i, k in zip(instructions, keep, strict=True)]
        else:
            demos = None
            demo_labels = None

        inputs: dict[str, Any] = {"images": images, "instructions": instructions}
        if demos is not None:
            inputs["demos"] = demos
            inputs["demo_labels"] = demo_labels

        # Videos [B, V, T, C, H, W] - only assembled during training when the world model consumes them.
        if self.model.config.enable_world_model and training:
            views = [batch[k].unsqueeze(1) if batch[k].ndim == 4 else batch[k] for k in image_keys]
            # A single stacked [B, V, T, C, H, W] tensor needs one spatial size for every view, and
            # cameras can differ (base 480x640 vs wrist 720x1280). Resize to `resize_images_to`, else
            # to the first view's size (a no-op for single-resolution datasets). The vjepa video
            # processor handles the final resize to the encoder resolution.
            cfg = self.model.config
            target_hw = tuple(cfg.resize_images_to) if cfg.resize_images_to else tuple(views[0].shape[-2:])
            resized = []
            for v in views:
                if tuple(v.shape[-2:]) != target_hw:
                    b, t, c = v.shape[0], v.shape[1], v.shape[2]
                    v = F.interpolate(
                        v.reshape(b * t, c, v.shape[3], v.shape[4]).float(),
                        size=target_hw,
                        mode="bilinear",
                        align_corners=False,
                    ).reshape(b, t, c, target_hw[0], target_hw[1])
                resized.append(v)
            inputs["videos"] = self.model.qwen.to_pixel_values(torch.stack(resized, dim=1))

        actions = batch.get(ACTION)
        if actions is not None:
            inputs["actions"] = (actions.unsqueeze(1) if actions.ndim == 2 else actions).float()
            if (pad := batch.get("action_is_pad")) is not None:
                inputs["action_is_pad"] = pad

        state = batch.get(OBS_STATE)
        if state is not None:
            if state.ndim > 2:
                # deltas are forward-looking here, so index 0 is the current observation, not -1.
                state = state[:, 0, :]
            inputs["state"] = (state.unsqueeze(1) if state.ndim == 2 else state).float()  # [B, 1, dim]

        return inputs

    # ---- LeRobot Policy Interface ----

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """LeRobot train forward: convert → native forward → aggregate losses."""
        native_output = self.model.forward(
            **self._prepare_model_inputs(batch, training=True), reduction=reduction
        )

        ref = next(iter(native_output.values()))
        zero = torch.zeros_like(ref)
        total_loss = (
            native_output.get("action_loss", zero)
            + native_output.get("wm_loss", zero)
            + native_output.get("demo_contrastive_loss", zero)
        )
        logs = {k: v.detach().mean().item() for k, v in native_output.items()}
        logs["loss"] = total_loss.detach().mean().item()
        return total_loss, logs

    def get_optim_params(self) -> dict:
        return self.model.parameters()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """LeRobot inference: convert → native predict → return as Tensor."""
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        inputs = self._prepare_model_inputs(batch, training=False)
        actions = self.model.predict_action(
            inputs["images"], inputs["instructions"], inputs.get("state"), inputs.get("demos")
        )
        return actions.to(device=self.config.device, dtype=torch.float32)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """LeRobot select_action with action queue caching."""
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        return self._queues[ACTION].popleft()

    @classmethod
    def _load_as_safetensor(cls, model: T, model_file: str, map_location: str, strict: bool) -> T:
        reinit_prefixes = model.config.reinit_modules
        if not reinit_prefixes:
            return super()._load_as_safetensor(model, model_file, map_location, strict)

        # `resolve_safetensors_device` is what keeps every rank from materializing the whole
        # checkpoint on GPU 0: safetensors maps the bare string "cuda" to cuda:0 regardless of
        # torch.cuda.current_device(), and `config.device` is exactly that bare string.
        state_dict = load_file(model_file, device=resolve_safetensors_device(map_location))
        current = model.state_dict()

        reinitialized: list[str] = []
        filtered: dict = {}
        for key, value in state_dict.items():
            if key in current and value.shape != current[key].shape:
                if not any(key.startswith(p) for p in reinit_prefixes):
                    raise ValueError(
                        f"Shape mismatch for '{key}' (checkpoint {tuple(value.shape)} vs model "
                        f"{tuple(current[key].shape)}) and its prefix is not in `reinit_modules`."
                    )
                reinitialized.append(
                    f"{key}: checkpoint {tuple(value.shape)} → model {tuple(current[key].shape)}"
                )
            else:
                filtered[key] = value

        if reinitialized:
            logging.warning(
                f"reinit_modules: skipping {len(reinitialized)} tensor(s) with mismatched shapes "
                f"(randomly re-initialised):\n  " + "\n  ".join(reinitialized)
            )

        # Deliberately non-strict: the reinitialized tensors above are *expected* to be missing.
        # `strict` still has to mean something, so enforce it on everything else.
        missing_keys, unexpected_keys = model.load_state_dict(filtered, strict=False)
        if strict:
            reinit_keys = {entry.split(":", 1)[0] for entry in reinitialized}
            unaccounted = [k for k in missing_keys if k not in reinit_keys]
            if unaccounted or unexpected_keys:
                raise RuntimeError(
                    f"Error(s) in loading state_dict for {type(model).__name__} with strict=True: "
                    f"missing keys not covered by `reinit_modules` {unaccounted}, "
                    f"unexpected keys {list(unexpected_keys)}."
                )
        log_model_loading_keys(missing_keys, unexpected_keys)
        return model
