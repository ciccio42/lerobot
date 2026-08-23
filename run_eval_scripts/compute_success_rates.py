#!/usr/bin/env python3
"""Recompute LIBERO success rates from an `eval_info.json` produced by `lerobot-eval`.

Reads the raw per-episode `successes` arrays under `per_task` and independently
recomputes the per-task and per-suite (`per_group`) success rates, then compares
the per-suite and overall results against the `pc_success` values `lerobot-eval`
already wrote to `per_group` / `overall`, to check the two agree.

Usage:
    python compute_success_rates.py [--eval-info PATH]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

DEFAULT_EVAL_INFO = Path(
    "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/lerobot/lerobot/eval_logs/pi0_libero/eval_info.json"
)


def compute_success_rates(data: dict) -> tuple[dict, dict]:
    """Recompute per-task and per-suite success rates from raw `successes` arrays."""
    per_task_rates = {}
    group_successes = defaultdict(list)

    for task in data["per_task"]:
        group = task["task_group"]
        task_id = task["task_id"]
        successes = task["metrics"]["successes"]
        n_success = sum(successes)
        per_task_rates[(group, task_id)] = {
            "n_episodes": len(successes),
            "n_success": n_success,
            "pc_success": 100.0 * n_success / len(successes),
        }
        group_successes[group].extend(successes)

    per_group_rates = {}
    for group, successes in group_successes.items():
        n_success = sum(successes)
        per_group_rates[group] = {
            "n_episodes": len(successes),
            "n_success": n_success,
            "pc_success": 100.0 * n_success / len(successes),
        }

    return per_task_rates, per_group_rates


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-info", type=Path, default=DEFAULT_EVAL_INFO)
    args = parser.parse_args()

    data = json.loads(args.eval_info.read_text())
    per_task_rates, per_group_rates = compute_success_rates(data)

    print("=" * 78)
    print("Per-task success rates (computed from raw `successes` arrays)")
    print("=" * 78)
    for (group, task_id), stats in sorted(per_task_rates.items()):
        print(
            f"  {group:<16} task {task_id:>2}: "
            f"{stats['n_success']:>2}/{stats['n_episodes']:<3} = {stats['pc_success']:6.2f}%"
        )
    # Note: eval_info.json has no per-task `pc_success` to compare against, only
    # the raw `successes` list, so there's nothing to cross-check at this level.

    print()
    print("=" * 78)
    print("Per-suite success rates: computed (from `per_task`) vs. reported (`per_group`)")
    print("=" * 78)
    all_match = True
    for group, stats in sorted(per_group_rates.items()):
        reported = data["per_group"][group]["pc_success"]
        computed = stats["pc_success"]
        match = abs(computed - reported) < 1e-6
        all_match &= match
        print(
            f"  {group:<16} computed={computed:6.2f}%  reported={reported:6.2f}%  "
            f"[{'OK' if match else 'MISMATCH'}]"
        )

    print()
    print("=" * 78)
    print("Overall success rate: computed (from `per_task`) vs. reported (`overall`)")
    print("=" * 78)
    total_success = sum(s["n_success"] for s in per_group_rates.values())
    total_episodes = sum(s["n_episodes"] for s in per_group_rates.values())
    overall_computed = 100.0 * total_success / total_episodes
    overall_reported = data["overall"]["pc_success"]
    match = abs(overall_computed - overall_reported) < 1e-6
    all_match &= match
    print(
        f"  overall ({total_success}/{total_episodes}) computed={overall_computed:6.2f}%  "
        f"reported={overall_reported:6.2f}%  [{'OK' if match else 'MISMATCH'}]"
    )

    print()
    if all_match:
        print("All computed success rates match the values reported in eval_info.json.")
    else:
        print("MISMATCH detected between computed and reported success rates -- see above.")


if __name__ == "__main__":
    main()
