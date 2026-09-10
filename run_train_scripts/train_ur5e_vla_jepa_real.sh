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

# Real-world counterpart to train_ur5e_vla_jepa.sh: fine-tune VLA-JEPA on
# real_ur5e_pick_place_delta_removed_0_5_10_15_lerobot (480 episodes / 25704
# frames, real ZED-camera UR5e trajectories, same 12 train_tasks / same
# held-out 0/5/10/15 split, same feature schema - front/gripper images, 13D
# observation.state, 7D action - as the sim dataset, both built by
# ../../convert_dataset.py from the same UR5E_FEATURES).
#
# Unlike train_ur5e_vla_jepa.sh (which starts from the generic DROID-trained
# `lerobot/VLA-JEPA-Pretrain`), this warm-starts from THIS REPO'S OWN sim
# UR5e checkpoint (ur5e_vla_jepa_ur5e_pick_place_delta_removed_0_5_10_15,
# step 32000 - already fine-tuned + rollout-evaluated on the matching sim
# task split) - mirroring the osvi-awda real finetune's sim->real warm-start
# recipe (mtlfd_adaptation/train_mtlfd_real_eye_in_hand_ur5e.sh). Since that
# checkpoint's action/state schema already exactly matches this real
# dataset's (identical UR5E_FEATURES), --policy.reinit_modules is dropped -
# unlike the from-DROID sim run, no dimensionality mismatch exists to force
# reinitializing the action/state encoder-decoder. --rename_map is still
# required: the checkpoint's own weights/processors were themselves trained
# under the exterior_1_left/exterior_2_left names (inherited from ITS OWN
# DROID warm start) - see that checkpoint's train_config.json
# policy.input_features - so the same rename is needed regardless of which
# checkpoint we start from.
#
# gpuq's hard walltime cap is 7h, so this script is resumable exactly like
# train_ur5e_vla_jepa.sh - see that script's comments for the full mechanism
# (checkpoints/last detection, --steps/--scheduler.num_decay_steps bump on
# resume, SIGTERM self-resubmission chain).

set -uo pipefail

SCRIPT_PATH="/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/lerobot/lerobot/run_train_scripts/train_ur5e_vla_jepa_real.sh"

source /hpc/apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate lerobot

DATASET_REPO_ID=${DATASET_REPO_ID:-local/real_ur5e_pick_place_delta_removed_0_5_10_15}
DATASET_ROOT=${DATASET_ROOT:-/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/open_x_embodiment/datasets/real_ur5e_pick_place_delta_removed_0_5_10_15_lerobot}
PRETRAINED_PATH=${PRETRAINED_PATH:-/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/lerobot/lerobot/outputs/ur5e_vla_jepa_ur5e_pick_place_delta_removed_0_5_10_15/checkpoints/032000/pretrained_model}
JOB_NAME=${JOB_NAME:-ur5e_vla_jepa_real}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/${JOB_NAME}}
STEPS=${STEPS:-8000}
SAVE_FREQ=${SAVE_FREQ:-500}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_PROCESSES=${NUM_PROCESSES:-4}
NUM_WORKERS=${NUM_WORKERS:-8}

export HF_HOME=${HF_HOME:-/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-"${HF_HOME}/hub"}
mkdir -p "${HUGGINGFACE_HUB_CACHE}"

export HF_HUB_DISABLE_PROGRESS_BARS=1

cd "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/lerobot/lerobot"

RESUME_CONFIG="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"

STEP_BEFORE=$(basename "$(readlink -f "${OUTPUT_DIR}/checkpoints/last" 2>/dev/null)" 2>/dev/null | sed 's/^0*//')
STEP_BEFORE=${STEP_BEFORE:-0}

if [ -f "${RESUME_CONFIG}" ]; then
  echo "Found existing checkpoint, resuming from ${RESUME_CONFIG} with steps=${STEPS}"
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
    --policy.repo_id=local/ur5e_vla_jepa_real_finetune
    --policy.device=cuda
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
