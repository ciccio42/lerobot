#!/usr/bin/env python
"""Minimal HTTP server exposing a trained LeRobot policy for out-of-process inference.

Exists so robosuite_test (Python 3.9) can query a LeRobot policy (Python 3.12+) without
merging the two environments. Run inside the `lerobot` conda env on the same GPU node as
the robosuite_test client. See VLA-Benchmark/robosuite_test/LEROBOT_EVAL.md for the full plan.
"""
import argparse
import base64
import io
import os

import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor import TransitionKey

app = Flask(__name__)
STATE = {}


def decode_image(b64_png: str) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64_png))).convert("RGB"))


@app.get("/")
def health():
    return "ok"


@app.post("/predict")
def predict():
    payload = request.get_json()
    device = STATE["device"]

    batch = {"task": [payload["task_description"]]}
    for key, b64_png in payload["images"].items():
        img = decode_image(b64_png)  # HWC uint8
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        batch[f"observation.images.{key}"] = t.unsqueeze(0).to(device)

    state = torch.tensor(payload["state"], dtype=torch.float32).unsqueeze(0).to(device)
    batch["observation.state"] = state

    obs = STATE["preprocessor"](batch)
    with torch.no_grad():
        action = STATE["policy"].select_action(obs)
    action = STATE["postprocessor"](action)
    action = action.squeeze(0).cpu().numpy().tolist()

    if STATE.get("flip_gripper_sign"):
        gripper_dim = STATE["gripper_dim"]
        action[gripper_dim] = -action[gripper_dim]

    # policy_type lets the client decide whether the gripper dim (last) needs the same
    # dataset-scale correction as position/orientation, or is already a directly-usable value
    # (e.g. VLA-JEPA's postprocessor already binarizes gripper to {-1, +1} independent of the
    # dataset's raw scale — see VLA-Benchmark/robosuite_test/models/lerobot_policy.py).
    return jsonify({"action": action, "policy_type": STATE["policy_type"]})  # 7-D: dx,dy,dz,droll,dpitch,dyaw,gripper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda")
    # MolmoAct2 refuses to run select_action() without this set explicitly (it has no default —
    # see configuration_molmoact2.py::resolve_inference_action_mode). Other policy types ignore it.
    parser.add_argument("--inference_action_mode", default="continuous", choices=["continuous", "discrete"])
    args = parser.parse_args()

    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    cfg.device = args.device
    if cfg.type == "molmoact2":
        cfg.inference_action_mode = args.inference_action_mode
    postprocessor_overrides = {}
    flip_gripper_sign = False
    if cfg.type == "vla_jepa":
        # VLA-JEPA's saved BinarizeGripperProcessorStep is broken for this dataset. It re-shares
        # `gripper_threshold` (0.5) with PreSnapGripperProcessorStep, which needs it in
        # *normalized* [-1,1] space, while Binarize applies it *after* UnnormalizerProcessorStep,
        # to real dataset units. This dataset's gripper action is MIN_MAX-normalized over a real
        # range of [0, 20], and PreSnapGripperProcessorStep's identity {0,1} decision unnormalizes
        # to real values ~10.0 (open) / ~20.0 (close) (empirically confirmed via
        # LEROBOT_DEBUG_GRIPPER below) — both far above 0.5, so `a > 0.5` is always true and
        # Binarize collapses every prediction to -1.0, regardless of what PreSnap decided.
        # PreSnap's own decision is fine on its own (observed ~51/49 open/close split over a real
        # rollout), so we override the saved threshold to 15.0 (the midpoint separating the two
        # real clusters) so Binarize actually distinguishes them again. Note `make_pre_post_processors`
        # loads the postprocessor from the checkpoint's saved JSON when `pretrained_path` is given,
        # so mutating `cfg` fields before this call has no effect — only `overrides=` reaches it.
        # This flips Binarize's own sign convention too though (`a > threshold -> -1.0`, so the
        # *high* real cluster (20, close) now maps to -1 and the *low* cluster (10, open) to +1)
        # — the opposite of what pick_place.py expects (+1 = close) — so we flip the sign back
        # below, after the postprocessor runs.
        postprocessor_overrides["vla_jepa_binarize_gripper"] = {"threshold": 15.0}
        flip_gripper_sign = True
    # make_policy() requires a ds_meta or env_cfg to (re)derive feature shapes — neither exists
    # here (no LeRobot dataset/env involved), so load directly via the policy class's own
    # from_pretrained instead; the checkpoint's saved config.json already has resolved
    # input/output features baked in.
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(args.policy_path, config=cfg)
    policy.to(args.device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=args.policy_path, postprocessor_overrides=postprocessor_overrides
    )

    # Temporary diagnostic: log the gripper action value after every postprocessor step, so we
    # can see the raw (pre-binarize) unnormalized gripper prediction instead of only the final
    # binarized {-1, +1}. Gated behind an env var, no effect unless explicitly enabled — see
    # VLA-Benchmark/robosuite_test/LEROBOT_EVAL.md investigation into VLA-JEPA never closing.
    if os.environ.get("LEROBOT_DEBUG_GRIPPER"):
        gripper_dim = getattr(cfg, "gripper_dim", 6)

        def _log_gripper_step(idx, transition):
            action = transition.get(TransitionKey.ACTION)
            if action is not None and action.shape[-1] > gripper_dim:
                step_name = postprocessor.steps[idx].__class__.__name__
                val = action[..., gripper_dim].flatten()[0].item()
                print(f"[GRIPPER_DEBUG] after step {idx} ({step_name}): {val:.4f}")

        postprocessor.after_step_hooks.append(_log_gripper_step)

    STATE.update(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=args.device,
        policy_type=cfg.type,
        gripper_dim=getattr(cfg, "gripper_dim", 6),
        flip_gripper_sign=flip_gripper_sign,
    )
    print(f"LeRobot policy server ready on port {args.port} (policy_path={args.policy_path})")
    app.run(host="127.0.0.1", port=args.port, threaded=False)


if __name__ == "__main__":
    main()
