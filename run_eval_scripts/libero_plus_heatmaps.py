"""Gripper XY heat-maps from LIBERO(-plus) evaluation rollouts, in the same style/frame as
LIBERO-plus/command_xy_heatmaps (LIBERO-plus/dataset_analysis.py -> utils.save_command_heatmaps).

Input: the per-episode pickles written by lerobot-eval with LEROBOT_EVAL_SAVE_TRAJ=1
  <eval_dir>/trajectories/<suite>_<task_id>/eval_episode_<i>.pkl
  (keys: eef_pos (T+1,3) world frame, eef_quat, gripper_qpos, joint_pos, actions, success, task_description)
eval_dir can be a single shard or any parent (e.g. eval_logs/libero_plus/vla_jepa): all pickles below it
are used.

--group_by
  command   one map per language instruction (as the dataset heat-maps; note that "Language
            Instructions" variants paraphrase the command, so they get their own maps)
  task      one map per original LIBERO task (perturbation suffix stripped), all variants pooled
  category  one map per original task x LIBERO-plus perturbation category (task_classification.json)
--outcome all | success | failure   filter episodes by outcome

Usage (lerobot env; --group_by task/category also need the LIBERO-plus overlay on PYTHONPATH):
  O=/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/lerobot/libero_plus_overlay
  PYTHONPATH=$O:$O/site LIBERO_CONFIG_PATH=$O/config python libero_plus_heatmaps.py \
      --eval_dir ../eval_logs/libero_plus/vla_jepa --group_by category --out_dir heatmaps/vla_jepa
"""
import argparse
import collections
import glob
import json
import os
import pickle
import re
import sys

import numpy as np

LIBERO_PLUS = "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/LIBERO-plus"
CLASSIFICATION = f"{LIBERO_PLUS}/libero/libero/benchmark/task_classification.json"
# perturbation suffixes of LIBERO-plus variant names (layout add/level, camera, light, language, table, robot init, noise)
SUFFIX_RE = re.compile(r"_(?:add_\d+|level\d+_sample\d+|view_.*|light_.*|language_.*|table_\d+|tb_\d+|initstate_.*|noise_.*)$")


def load_episodes(eval_dir, outcome):
    episodes = []
    for f in sorted(glob.glob(os.path.join(eval_dir, "**", "trajectories", "*", "eval_episode_*.pkl"), recursive=True)):
        with open(f, "rb") as fh:
            ep = pickle.load(fh)
        if outcome == "success" and not ep["success"] or outcome == "failure" and ep["success"]:
            continue
        suite, task_id = os.path.basename(os.path.dirname(f)).rsplit("_", 1)
        ep["suite"], ep["task_id"] = suite, int(task_id)
        episodes.append(ep)
    return episodes


def variant_names(episodes):
    """(suite, task_id) -> (variant name, category), through the LIBERO-plus benchmark."""
    from libero.libero import benchmark

    classes = json.load(open(CLASSIFICATION))
    bench = benchmark.get_benchmark_dict()
    out = {}
    for suite in {e["suite"] for e in episodes}:
        task_suite = bench[suite]()
        cat = {e["name"]: e["category"] for e in classes.get(suite, [])}
        for e in episodes:
            if e["suite"] == suite:
                name = task_suite.get_task(e["task_id"]).name
                out[(suite, e["task_id"])] = (name, cat.get(name, "unknown"))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--group_by", choices=["command", "task", "category"], default="command")
    p.add_argument("--outcome", choices=["all", "success", "failure"], default="all")
    p.add_argument("--min_points", type=int, default=10)
    args = p.parse_args()

    episodes = load_episodes(args.eval_dir, args.outcome)
    print(f"{len(episodes)} episodes ({args.outcome})")
    names = variant_names(episodes) if args.group_by != "command" else {}

    groups = collections.defaultdict(list)
    for e in episodes:
        if args.group_by == "command":
            key = e["task_description"]
        else:
            name, category = names[(e["suite"], e["task_id"])]
            base = SUFFIX_RE.sub("", name)
            key = base if args.group_by == "task" else f"{base} [{category}]"
        groups[key].append(np.asarray(e["eef_pos"])[:, :2])
    command_to_xy = {k: np.concatenate(v, axis=0) for k, v in groups.items()}

    sys.path.insert(0, LIBERO_PLUS)
    from utils import save_command_heatmaps  # LIBERO-plus/utils.py (same style as the dataset heat-maps)

    n = save_command_heatmaps(command_to_xy_points=command_to_xy, output_dir=args.out_dir, min_points=args.min_points)
    print(f"{n} heat-maps -> {args.out_dir}")
    json.dump({k: len(v) for k, v in groups.items()}, open(os.path.join(args.out_dir, "episodes_per_map.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
