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
#SBATCH --output=slurm_logs/%x-%j.out

# Fine-tune pi0.5 (lerobot/pi05_base) on a UR5e pick-place LeRobotDataset v3 — the pi0.5
# counterpart of train_ur5e_vla_jepa.sh. One script serves every training session: the
# session is selected via env vars (DATASET_REPO_ID / DATASET_ROOT / JOB_NAME /
# PRETRAINED_PATH); see launch_ur5e_pi05_sessions.sh for the full session list.
#
# Checkpoints are saved at the END OF EVERY EPOCH: SAVE_FREQ is set to the number of optimizer
# steps in one pass over the dataset, ceil(total_frames / (BATCH_SIZE * NUM_PROCESSES)), and
# STEPS = NUM_EPOCHS * SAVE_FREQ, so checkpoints/<step> land exactly on epoch boundaries
# (checkpoint k*SAVE_FREQ == end of epoch k). Note: when a chain link resumes, the dataloader
# restarts with a fresh shuffle, so "epoch k" means "k datasets' worth of samples seen", not a
# strict partition of the data.
#
# Camera mapping: pi05_base declares observation.images.{base_0_rgb,left_wrist_0_rgb,
# right_wrist_0_rgb}; our dataset has observation.images.{front,gripper}. We rename
# front->base_0_rgb and gripper->left_wrist_0_rgb; the missing right_wrist_0_rgb is filled by
# pi0.5 itself with an empty, masked-out image (modeling_pi05.py, missing_img_keys). State (13D)
# and action (7D) are zero-padded to max_state_dim/max_action_dim=32 by the policy, and both are
# QUANTILES-normalized using the q01/q99 stats already present in each dataset's meta/stats.json.
#
# Resumable + self-chaining past gpuq's 7h walltime cap, exactly like train_ur5e_vla_jepa.sh —
# see that script for the full rationale behind each piece (hardcoded SCRIPT_PATH, SIGTERM trap,
# "resubmit only if the checkpoint step advanced").
#
# Optional FOLLOWUP_SCRIPT: a script run (with bash) once this session's chain reaches its final
# step — used to start the real-robot session only after its sim warm-start checkpoint exists.

set -uo pipefail

SCRIPT_PATH="/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/lerobot/lerobot/run_train_scripts/train_ur5e_pi05.sh"

source /hpc/apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate lerobot

DATASET_REPO_ID=${DATASET_REPO_ID:-local/ur5e_pick_place_delta_all}
DATASET_ROOT=${DATASET_ROOT:-/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/open_x_embodiment/datasets/ur5e_pick_place_delta_all_lerobot}
PRETRAINED_PATH=${PRETRAINED_PATH:-lerobot/pi05_base}
JOB_NAME=${JOB_NAME:-ur5e_pi05}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/${JOB_NAME}}
NUM_EPOCHS=${NUM_EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-8}          # per GPU
NUM_PROCESSES=${NUM_PROCESSES:-4}
NUM_WORKERS=${NUM_WORKERS:-8}
FOLLOWUP_SCRIPT=${FOLLOWUP_SCRIPT:-}

export HF_HOME=${HF_HOME:-/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-"${HF_HOME}/hub"}
mkdir -p "${HUGGINGFACE_HUB_CACHE}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
# Everything needed (lerobot/pi05_base, google/paligemma-3b-pt-224 tokenizer) is already in the
# local cache. Two fresh starts (jobs 570990, 570992) hung with ranks 1-3 never finishing
# make_policy while all 4 ranks were making unauthenticated HF Hub requests; offline mode removes
# that network dependency entirely.
export HF_HUB_OFFLINE=1

cd "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/lerobot/lerobot"
mkdir -p slurm_logs

# Steps per epoch from the dataset's own frame count (accelerate shards each global batch of
# BATCH_SIZE*NUM_PROCESSES samples across the processes; drop_last=False -> ceil).
TOTAL_FRAMES=$(python -c "import json; print(json.load(open('${DATASET_ROOT}/meta/info.json'))['total_frames'])")
SAVE_FREQ=$(( (TOTAL_FRAMES + BATCH_SIZE * NUM_PROCESSES - 1) / (BATCH_SIZE * NUM_PROCESSES) ))
STEPS=$(( NUM_EPOCHS * SAVE_FREQ ))
echo "Session ${JOB_NAME}: ${TOTAL_FRAMES} frames, effective batch $((BATCH_SIZE * NUM_PROCESSES)) -> ${SAVE_FREQ} steps/epoch, ${NUM_EPOCHS} epochs = ${STEPS} steps"

RESUME_CONFIG="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"

STEP_BEFORE=$(basename "$(readlink -f "${OUTPUT_DIR}/checkpoints/last" 2>/dev/null)" 2>/dev/null | sed 's/^0*//')
STEP_BEFORE=${STEP_BEFORE:-0}

if [ -f "${RESUME_CONFIG}" ]; then
  echo "Found existing checkpoint, resuming from ${RESUME_CONFIG} with steps=${STEPS}"
  TRAIN_ARGS=(--config_path="${RESUME_CONFIG}" --resume=true --steps="${STEPS}" --save_freq="${SAVE_FREQ}" --wandb.disable_artifact=true)
else
  if [ -d "${OUTPUT_DIR}" ]; then
    echo "No checkpoint but ${OUTPUT_DIR} exists (stale dir from a crashed run) — clearing it before a fresh start"
    rm -rf "${OUTPUT_DIR}"
  fi
  echo "No existing checkpoint, starting fresh run from ${PRETRAINED_PATH}"
  TRAIN_ARGS=(
    --dataset.repo_id="${DATASET_REPO_ID}"
    --dataset.root="${DATASET_ROOT}"
    --dataset.video_backend=pyav
    --dataset.image_transforms.enable=true
    --policy.path="${PRETRAINED_PATH}"
    --policy.repo_id="local/${JOB_NAME}_finetune"
    --policy.device=cuda
    --policy.dtype=bfloat16
    --policy.gradient_checkpointing=true
    --policy.freeze_vision_encoder=false
    --policy.train_expert_only=false
    --policy.scheduler_decay_steps="${STEPS}"
    --policy.push_to_hub=false
    --rename_map='{"observation.images.front":"observation.images.base_0_rgb","observation.images.gripper":"observation.images.left_wrist_0_rgb"}'
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

resubmit_if_progressed() {
  local step_after
  step_after=$(basename "$(readlink -f "${OUTPUT_DIR}/checkpoints/last" 2>/dev/null)" 2>/dev/null | sed 's/^0*//')
  step_after=${step_after:-0}
  if [ "${step_after}" -gt "${STEP_BEFORE}" ] && [ "${step_after}" -lt "${STEPS}" ]; then
    echo "Progressed ${STEP_BEFORE} -> ${step_after} (target ${STEPS} not yet reached) — submitting the next chain job"
    sbatch --job-name="${JOB_NAME}" "${SCRIPT_PATH}" || echo "Self-resubmission failed (possibly at the cluster's MaxSubmit cap) — needs manual or external resubmission"
  elif [ "${step_after}" -ge "${STEPS}" ]; then
    echo "Reached target step ${STEPS} — chain complete"
    if [ -n "${FOLLOWUP_SCRIPT}" ]; then
      echo "Running follow-up ${FOLLOWUP_SCRIPT}"
      FOLLOWUP_SCRIPT= bash "${FOLLOWUP_SCRIPT}" || echo "Follow-up submission failed — run ${FOLLOWUP_SCRIPT} manually"
    fi
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

# srun runs in the background + `wait`: bash only runs a trap once the foreground command
# returns, so with a foreground srun that outlived SLURM's KillWait (job 570989), bash was
# SIGKILLed before on_term could resubmit the chain. `wait` returns as soon as SIGTERM arrives.
srun accelerate launch \
    --num_processes="${NUM_PROCESSES}" \
    --mixed_precision=bf16 \
    -m lerobot.scripts.lerobot_train \
    "${TRAIN_ARGS[@]}" &
wait $!
TRAIN_EXIT=$?

resubmit_if_progressed

exit "${TRAIN_EXIT}"
