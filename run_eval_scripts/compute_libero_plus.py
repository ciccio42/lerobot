"""Aggregates the LIBERO-plus shards (eval_logs/libero_plus/<model>/<suite>/<start>_<end>/eval_info.json)
into success rates per model x suite, perturbation dimension and difficulty level, using LIBERO-plus's
task_classification.json (variants are matched by task name through the fork's benchmark).

Run with the LIBERO-plus overlay on the path:
  O=/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/lerobot/libero_plus_overlay
  PYTHONPATH=$O:$O/site LIBERO_CONFIG_PATH=$O/config \
    /mnt/beegfs/frosa/.conda/envs/lerobot/bin/python compute_libero_plus.py
Writes eval_logs/libero_plus/summary.json and per_variant.csv.
"""
import collections
import csv
import glob
import json
import os

from libero.libero import benchmark

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval_logs", "libero_plus")
CLASSIFICATION = "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/LIBERO-plus/libero/libero/benchmark/task_classification.json"
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def main():
    classes = json.load(open(CLASSIFICATION))
    bench = benchmark.get_benchmark_dict()
    rows = []
    for suite in SUITES:
        by_name = {e["name"]: e for e in classes[suite]}
        task_suite = bench[suite]()
        for model in sorted(os.listdir(ROOT)):
            if model in ("slurm", "smoke") or not os.path.isdir(os.path.join(ROOT, model)):
                continue
            for f in sorted(glob.glob(os.path.join(ROOT, model, suite, "*", "eval_info.json"))):
                for t in json.load(open(f))["per_task"]:
                    tid = int(t["task_id"])
                    name = task_suite.get_task(tid).name
                    meta = by_name.get(name, {})
                    for success in t["metrics"]["successes"]:
                        rows.append({"model": model, "suite": suite, "task_id": tid, "name": name,
                                     "category": meta.get("category", "unknown"),
                                     "difficulty": meta.get("difficulty_level", -1), "success": int(success)})

    def rate(keys):
        acc = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            k = tuple(r[x] for x in keys)
            acc[k][0] += r["success"]
            acc[k][1] += 1
        return {" | ".join(map(str, k)): {"success_rate": round(100 * s / n, 2), "n": n} for k, (s, n) in sorted(acc.items())}

    summary = {
        "overall": rate(["model"]),
        "per_suite": rate(["model", "suite"]),
        "per_category": rate(["model", "category"]),
        "per_difficulty": rate(["model", "difficulty"]),
        "expected_variants_per_model": sum(len(classes[s]) for s in SUITES),
    }
    json.dump(summary, open(os.path.join(ROOT, "summary.json"), "w"), indent=2)
    with open(os.path.join(ROOT, "per_variant.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["model"])
        w.writeheader()
        w.writerows(rows)
    for section in ["overall", "per_suite", "per_category"]:
        print(f"== {section}")
        for k, v in summary[section].items():
            print(f"  {k:45s} {v['success_rate']:6.2f}%  (n={v['n']})")


if __name__ == "__main__":
    main()
