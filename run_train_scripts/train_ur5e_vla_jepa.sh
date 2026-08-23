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

set -euo pipefail

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

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/.."

RESUME_CONFIG="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"

if [ -f "${RESUME_CONFIG}" ]; then
  echo "Found existing checkpoint, resuming from ${RESUME_CONFIG}"
  TRAIN_ARGS=(--config_path="${RESUME_CONFIG}" --resume=true)
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
  )
fi

srun accelerate launch \
    --num_processes="${NUM_PROCESSES}" \
    --mixed_precision=bf16 \
    -m lerobot.scripts.lerobot_train \
    "${TRAIN_ARGS[@]}"
