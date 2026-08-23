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

set -euo pipefail

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
  )
fi

srun accelerate launch \
    --num_processes="${NUM_PROCESSES}" \
    --mixed_precision=bf16 \
    -m lerobot.scripts.lerobot_train \
    "${TRAIN_ARGS[@]}"
