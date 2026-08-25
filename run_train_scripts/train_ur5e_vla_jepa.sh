#!/bin/bash
#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:50:00
#SBATCH --export=ALL
#SBATCH --exclude=gnode09

# Fine-tune VLA-JEPA on the UR5e pick-place dataset (already converted to
# LeRobotDataset v3 by ../../convert_dataset.py). Starts from
# `lerobot/VLA-JEPA-Pretrain` (DROID-trained, the most embodiment-neutral of
# the three released checkpoints) per docs/source/vla_jepa.mdx's
# "Fine-tuning on a different embodiment" recipe. UR5e's
# `observation.state` is 13D vs the checkpoint's 8D, and even though our
# action_dim resolves to 7 (numerically matching), the action encoder/decoder
# are reinitialized too since the pretrained ones encode DROID action
# semantics, not UR5e's.
#
# gpuq's hard walltime cap is 7h (see partition MaxTime), so this script is
# resumable: it checks for an existing `checkpoints/last` and, if present,
# resumes from it instead of starting over. Submit a short chain of these
# jobs with `--dependency=afterany` to train past a single job's time limit;
# a job that finds training already complete will resume and exit quickly.
#
# VLA-JEPA has no `--policy.image_keys`-style override (unlike MolmoAct2):
# it matches dataset camera feature names against the checkpoint's expected
# ones exactly. `VLA-JEPA-Pretrain` (DROID) expects
# observation.images.{exterior_1_left,exterior_2_left}; our dataset has
# observation.images.{front,gripper}, so a --rename_map is required or
# training fails immediately with "Feature mismatch between dataset/
# environment and policy config".

set -uo pipefail

# Hardcoded, not derived from BASH_SOURCE/$0/SLURM_SUBMIT_DIR: SLURM runs a self-resubmitted
# batch job from a spooled copy of the script (e.g. /cm/local/.../spool/job<id>/slurm_script),
# so BASH_SOURCE[0] at runtime reflects that spool path, not this file's real location — every
# generation's `sbatch "${SCRIPT_PATH}"` would otherwise hand the *next* generation a
# throwaway spool path (confirmed: this previously caused multiple silent "no existing
# checkpoint, starting fresh" restarts once cwd/OUTPUT_DIR resolution drifted off this path,
# discarding real training progress each time). A hardcoded absolute path is immune to this
# regardless of how many generations deep the chain is.
SCRIPT_PATH="/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/lerobot/lerobot/run_train_scripts/train_ur5e_vla_jepa.sh"

source /hpc/apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate lerobot

DATASET_REPO_ID=${DATASET_REPO_ID:-local/ur5e_pick_place_delta_all}
DATASET_ROOT=${DATASET_ROOT:-/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/open_x_embodiment/datasets/ur5e_pick_place_delta_all_lerobot}
PRETRAINED_PATH=${PRETRAINED_PATH:-lerobot/VLA-JEPA-Pretrain}
JOB_NAME=${JOB_NAME:-ur5e_vla_jepa}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/${JOB_NAME}}
STEPS=${STEPS:-8000}
SAVE_FREQ=${SAVE_FREQ:-500}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_PROCESSES=${NUM_PROCESSES:-4}
NUM_WORKERS=${NUM_WORKERS:-8}

export HF_HOME=${HF_HOME:-/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-"${HF_HOME}/hub"}
mkdir -p "${HUGGINGFACE_HUB_CACHE}"

# Every resume reloads the base/checkpoint weights from scratch, each time re-emitting
# transformers' `tqdm(..., desc="Loading weights")` bar (src: core_model_loading.py) into the
# wandb-captured output log. transformers' own tqdm wrapper is gated by
# huggingface_hub's are_progress_bars_disabled(), which reads this env var — set before the
# interpreter starts, since it's checked once at transformers.utils.logging import time.
export HF_HUB_DISABLE_PROGRESS_BARS=1

# Hardcoded for the same reason SCRIPT_PATH is above — see that comment.
cd "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/lerobot/lerobot"

RESUME_CONFIG="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"

STEP_BEFORE=$(basename "$(readlink -f "${OUTPUT_DIR}/checkpoints/last" 2>/dev/null)" 2>/dev/null | sed 's/^0*//')
STEP_BEFORE=${STEP_BEFORE:-0}

if [ -f "${RESUME_CONFIG}" ]; then
  echo "Found existing checkpoint, resuming from ${RESUME_CONFIG} with steps=${STEPS}"
  # --steps overrides the resumed run's target step count (the scheduler recomputes its
  # warmup/decay horizon from this value at build time, so extending it properly "reheats" the
  # LR for genuine further training instead of continuing at an already-decayed-to-floor rate —
  # see LEROBOT_EVAL.md's 2026-08-23 training-resume investigation).
  #
  # --scheduler.num_decay_steps must ALSO be bumped to match: CosineDecayWithWarmupSchedulerConfig
  # only auto-scales its warmup/decay horizon *down* to fit num_training_steps when the configured
  # num_decay_steps is larger than the new step target (see optim/schedulers.py) — it does nothing
  # when steps is being extended *past* the saved num_decay_steps (originally 30000, from the
  # first fresh-start run). Without this override, extending past step 30000 would decay to the
  # LR floor there and then coast at that floor for the rest of the run, wasting most of any
  # extension beyond 30000 steps.
  TRAIN_ARGS=(--config_path="${RESUME_CONFIG}" --resume=true --steps="${STEPS}" --scheduler.num_decay_steps="${STEPS}" --wandb.disable_artifact=true)
else
  if [ -d "${OUTPUT_DIR}" ]; then
    echo "No checkpoint but ${OUTPUT_DIR} exists (stale dir from a crashed run) — clearing it before a fresh start"
    rm -rf "${OUTPUT_DIR}"
  fi
  echo "No existing checkpoint, starting fresh run"
  TRAIN_ARGS=(
    --dataset.repo_id="${DATASET_REPO_ID}"
    --dataset.root="${DATASET_ROOT}"
    --dataset.video_backend=pyav
    --dataset.image_transforms.enable=true
    --policy.path="${PRETRAINED_PATH}"
    --policy.repo_id=local/ur5e_vla_jepa_finetune
    --policy.device=cuda
    --policy.reinit_modules='["model.action_model.action_encoder", "model.action_model.action_decoder", "model.action_model.state_encoder"]'
    --policy.push_to_hub=false
    --rename_map='{"observation.images.front":"observation.images.exterior_1_left","observation.images.gripper":"observation.images.exterior_2_left"}'
    --output_dir="${OUTPUT_DIR}"
    --job_name="${JOB_NAME}"
    --steps="${STEPS}"
    --batch_size="${BATCH_SIZE}"
    --num_workers="${NUM_WORKERS}"
    --log_freq=20
    --eval_freq=-1
    --save_checkpoint=true
    --save_freq="${SAVE_FREQ}"
    --wandb.enable=true
    --wandb.project="${WANDB_PROJECT:-ur5e-finetune}"
    --wandb.disable_artifact=true
  )
fi

# Self-chaining: resubmit the next job in this chain from *inside* the job itself, on the
# compute node, rather than relying on an external submitter (a session-side watcher loop, or
# SLURM --dependency alone — dependencies only delay an *already-submitted* job, they don't
# create new ones). An external watcher dies with whatever process started it; this doesn't,
# since it runs as part of the SLURM job's own script.
#
# Resubmit criterion is "did the checkpoint step count actually advance", NOT "did the process
# exit 0" — the single most common reason to need the next link in this chain is gpuq's 7h
# walltime cap killing this job mid-training, and that case must still chain. Only refuse to
# resubmit when NO progress was made at all (STEP_AFTER <= STEP_BEFORE), which catches a real,
# immediate failure (e.g. crashes before its first checkpoint save) without an infinite
# resubmission loop burning the account's job-submit quota on a persistently broken run.
#
# This logic runs from BOTH a normal post-command flow AND a SIGTERM trap: when SLURM kills the
# job for hitting its walltime, it sends SIGTERM to the whole job — including this parent bash
# script, not just the srun'd training process — and an untrapped script dies immediately
# without ever reaching the lines after srun (confirmed: MolmoAct2's job 536599 timed out with
# checkpoint 30000 already on disk but never resubmitted, because it had no trap). KillWait
# (30s here) is the budget between SIGTERM and a hard SIGKILL, comfortably enough for the
# sbatch call below.
resubmit_if_progressed() {
  local step_after
  step_after=$(basename "$(readlink -f "${OUTPUT_DIR}/checkpoints/last" 2>/dev/null)" 2>/dev/null | sed 's/^0*//')
  step_after=${step_after:-0}
  if [ "${step_after}" -gt "${STEP_BEFORE}" ] && [ "${step_after}" -lt "${STEPS}" ]; then
    echo "Progressed ${STEP_BEFORE} -> ${step_after} (target ${STEPS} not yet reached) — submitting the next chain job"
    STEPS="${STEPS}" sbatch "${SCRIPT_PATH}" || echo "Self-resubmission failed (possibly at the cluster's MaxSubmit cap) — needs manual or external resubmission"
  elif [ "${step_after}" -ge "${STEPS}" ]; then
    echo "Reached target step ${STEPS} — chain complete"
  else
    echo "No progress this run (stuck at step ${step_after}) — NOT auto-resubmitting; investigate before continuing this chain manually"
  fi
}

on_term() {
  echo "Caught SIGTERM (likely gpuq's walltime limit) — checking progress before this job dies"
  resubmit_if_progressed
  exit 124
}
trap on_term TERM

srun accelerate launch \
    --num_processes="${NUM_PROCESSES}" \
    --mixed_precision=bf16 \
    -m lerobot.scripts.lerobot_train \
    "${TRAIN_ARGS[@]}"
TRAIN_EXIT=$?

resubmit_if_progressed

exit "${TRAIN_EXIT}"
