#!/usr/bin/env python
"""Minimal HTTP server exposing a trained LeRobot policy for out-of-process inference.

Exists so robosuite_test (Python 3.9) can query a LeRobot policy (Python 3.12+) without
merging the two environments. Run inside the `lerobot` conda env on the same GPU node as
the robosuite_test client. See VLA-Benchmark/robosuite_test/LEROBOT_EVAL.md for the full plan.
"""
import argparse
import base64
import io

import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

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
    # make_policy() requires a ds_meta or env_cfg to (re)derive feature shapes — neither exists
    # here (no LeRobot dataset/env involved), so load directly via the policy class's own
    # from_pretrained instead; the checkpoint's saved config.json already has resolved
    # input/output features baked in.
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(args.policy_path, config=cfg)
    policy.to(args.device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=args.policy_path
    )

    STATE.update(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=args.device,
        policy_type=cfg.type,
    )
    print(f"LeRobot policy server ready on port {args.port} (policy_path={args.policy_path})")
    app.run(host="127.0.0.1", port=args.port, threaded=False)


if __name__ == "__main__":
    main()
