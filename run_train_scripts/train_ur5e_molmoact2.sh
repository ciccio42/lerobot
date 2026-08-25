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

# Fine-tune MolmoAct2 on the UR5e pick-place dataset (already converted to
# LeRobotDataset v3 by ../../convert_dataset.py). Starts from the base
# `allenai/MolmoAct2` release weights (not the LIBERO-tuned variant, since
# the target embodiment/action-space is UR5e real-robot delta-EEF, not
# LIBERO sim) and LoRA-tunes the VLM while keeping the action expert fully
# trainable, per docs/source/molmoact2.mdx "Common Practices" guidance for
# a single-dataset real-world fine-tune.
#
# --policy.normalize_gripper=true (docs default is false): the masked
# passthrough path requires raw gripper actions already in [-1, 1] when
# false. This dataset's gripper action dim is a raw 0-20 range, not
# pre-scaled, so leaving the default crashes with "MolmoAct2 action gripper
# values are not under [-1, 1]" — true routes it through quantile
# normalization like every other action dim instead.
#
# gpuq's hard walltime cap is 7h (see partition MaxTime), so this script is
# resumable: it checks for an existing `checkpoints/last` and, if present,
# resumes from it instead of starting over. Submit a short chain of these
# jobs with `--dependency=afterany` to train past a single job's time limit;
# a job that finds training already complete will resume and exit quickly.

set -uo pipefail

# Hardcoded, not derived from BASH_SOURCE/$0/SLURM_SUBMIT_DIR: SLURM runs a self-resubmitted
# batch job from a spooled copy of the script (e.g. /cm/local/.../spool/job<id>/slurm_script),
# so BASH_SOURCE[0] at runtime reflects that spool path, not this file's real location — every
# generation's `sbatch "${SCRIPT_PATH}"` would otherwise hand the *next* generation a
# throwaway spool path (confirmed: this previously caused multiple silent "no existing
# checkpoint, starting fresh" restarts once cwd/OUTPUT_DIR resolution drifted off this path,
# discarding real training progress each time). A hardcoded absolute path is immune to this
# regardless of how many generations deep the chain is.
SCRIPT_PATH="/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/lerobot/lerobot/run_train_scripts/train_ur5e_molmoact2.sh"

source /hpc/apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate lerobot

DATASET_REPO_ID=${DATASET_REPO_ID:-local/ur5e_pick_place_delta_all}
DATASET_ROOT=${DATASET_ROOT:-/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/open_x_embodiment/datasets/ur5e_pick_place_delta_all_lerobot}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-allenai/MolmoAct2}
JOB_NAME=${JOB_NAME:-ur5e_molmoact2}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/${JOB_NAME}}
STEPS=${STEPS:-8000}
SAVE_FREQ=${SAVE_FREQ:-500}
BATCH_SIZE=${BATCH_SIZE:-8}
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
  TRAIN_ARGS=(--config_path="${RESUME_CONFIG}" --resume=true --steps="${STEPS}" --wandb.disable_artifact=true)
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
    --policy.type=molmoact2
    --policy.checkpoint_path="${CHECKPOINT_PATH}"
    --policy.device=cuda
    --policy.action_mode=continuous
    --policy.chunk_size=10
    --policy.n_action_steps=10
    --policy.setup_type="single UR5e robotic arm"
    --policy.control_mode="delta end-effector pose"
    --policy.image_keys='["observation.images.front","observation.images.gripper"]'
    --policy.model_dtype=bfloat16
    --policy.num_flow_timesteps=8
    --policy.gradient_checkpointing=true
    --policy.freeze_embedding=true
    --policy.normalize_gripper=true
    --policy.enable_lora_vlm=true
    --policy.push_to_hub=false
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
# without ever reaching the lines after srun (confirmed: job 536599 timed out with checkpoint
# 30000 already on disk but never resubmitted, because it had no trap). KillWait (30s here) is
# the budget between SIGTERM and a hard SIGKILL, comfortably enough for the sbatch call below.
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
