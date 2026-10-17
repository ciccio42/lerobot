#!/bin/bash
# Full LIBERO-plus evaluation (1 episode on each of the 10030 variants of libero_spatial/object/goal/10)
# for pi0.5 and VLA-JEPA, split into shards of run_libero_plus.sh. Runs on the login node and submits
# shards as SLURM submit slots free up (the account allows 10 submitted jobs), always leaving RESERVE
# slots free for other self-resubmitting chains (e.g. Interleave training). Shards that already have an
# eval_info.json are skipped, so the script can simply be started again; a failed shard is retried once.
#
# Usage (from anywhere):
#   nohup setsid bash run_libero_plus_all.sh > ../eval_logs/libero_plus/orchestrator.log 2>&1 &
# Env: MODELS ("pi05 vla_jepa"), SUITES (4 standard), SHARD_vla_jepa (300), SHARD_pi05 (150),
#      MAX_SUBMIT (10), RESERVE (1), POLL (120 s).
# Results: python compute_libero_plus.py  (per suite / perturbation dimension / difficulty)
set -uo pipefail
cd "$(dirname "$0")"
SCRIPT=$(pwd)/run_libero_plus.sh
LOGROOT=$(cd .. && pwd)/eval_logs/libero_plus
mkdir -p "${LOGROOT}/slurm"
MODELS=${MODELS:-"pi05 vla_jepa"}
SUITES=${SUITES:-"libero_spatial libero_object libero_goal libero_10"}
declare -A N_TASKS=([libero_spatial]=2402 [libero_object]=2518 [libero_goal]=2591 [libero_10]=2519)
declare -A SHARD=([vla_jepa]=${SHARD_vla_jepa:-300} [pi05]=${SHARD_pi05:-150})
MAX_SUBMIT=${MAX_SUBMIT:-10}
RESERVE=${RESERVE:-1}
POLL=${POLL:-120}

# shard list: model suite start end
SHARDS=()
for m in ${MODELS}; do
  for s in ${SUITES}; do
    for ((a = 0; a < ${N_TASKS[$s]}; a += ${SHARD[$m]})); do
      b=$((a + ${SHARD[$m]})); [ $b -gt ${N_TASKS[$s]} ] && b=${N_TASKS[$s]}
      SHARDS+=("$m $s $a $b")
    done
  done
done
echo "$(date '+%F %T') ${#SHARDS[@]} shards"

declare -A JOB TRIES
outdir() { echo "${LOGROOT}/$1/$2/$(printf '%04d_%04d' "$3" "$4")"; }
while true; do
  remaining=0
  for sh in "${SHARDS[@]}"; do
    read -r m s a b <<< "${sh}"
    od=$(outdir "$m" "$s" "$a" "$b")
    [ -f "${od}/eval_info.json" ] && continue
    remaining=$((remaining + 1))
    jid=${JOB[$sh]:-}
    if [ -z "${jid}" ]; then  # adopt a shard job submitted by a previous orchestrator instance
      jid=$(squeue -h -u "$USER" -n "lp_${m}_${s#libero_}_${a}" -o %i | head -1)
      if [ -n "${jid}" ]; then JOB[$sh]=${jid}; TRIES[$sh]=1; echo "$(date '+%F %T') adopted ${sh} -> job ${jid}"; fi
    fi
    if [ -n "${jid}" ] && squeue -h -j "${jid}" 2>/dev/null | grep -q .; then continue; fi
    if [ -n "${jid}" ]; then  # finished without eval_info.json
      echo "$(date '+%F %T') shard ${sh} (job ${jid}) ended without results ($(sacct -j "${jid}" -X -n -o State | head -1 | xargs))"
      JOB[$sh]=""
    fi
    t=${TRIES[$sh]:-0}
    if [ "$t" -ge 2 ]; then continue; fi
    n_jobs=$(squeue -h -u "$USER" | wc -l)
    if [ "${n_jobs}" -ge $((MAX_SUBMIT - RESERVE)) ]; then continue; fi
    out=$(MODEL=$m SUITE=$s TASK_START=$a TASK_END=$b sbatch --parsable --job-name="lp_${m}_${s#libero_}_${a}" \
          --output="${LOGROOT}/slurm/${m}_${s}_${a}_${b}_%j.out" "${SCRIPT}" 2>&1) || { echo "$(date '+%F %T') sbatch failed: ${out}"; continue; }
    JOB[$sh]=${out}; TRIES[$sh]=$((t + 1))
    echo "$(date '+%F %T') submitted ${sh} -> job ${out} (try $((t + 1)))"
  done
  if [ "${remaining}" -eq 0 ]; then echo "$(date '+%F %T') ALL DONE"; break; fi
  gave_up=$(for sh in "${SHARDS[@]}"; do read -r m s a b <<< "${sh}"; [ ! -f "$(outdir "$m" "$s" "$a" "$b")/eval_info.json" ] && [ "${TRIES[$sh]:-0}" -ge 2 ] && [ -z "${JOB[$sh]:-}" ] && echo x; done | wc -l)
  if [ "${gave_up}" -eq "${remaining}" ]; then echo "$(date '+%F %T') stopping: ${gave_up} shards failed twice"; break; fi
  sleep "${POLL}"
done
