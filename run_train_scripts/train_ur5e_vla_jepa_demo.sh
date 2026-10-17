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

# Demo-video-conditioned VLA-JEPA on UR5e pick-place (see VLA_JEPA_VIDEO_CONDITIONING.md).
# Same recipe as train_ur5e_vla_jepa.sh (VLA-JEPA-Pretrain warm start, rename_map, reinit of the
# action/state encoder-decoder, resumable self-chaining job), plus:
#   --policy.conditioning   demo | both   (both = demo + instruction, with LANGUAGE_DROPOUT)
#   --policy.demo_encoder   vjepa (option A: V-JEPA2 + resampler) | qwen_video (option C, baseline)
# The demo shown for each robot sample is a human video of the same task, looked up by the task
# instruction in DEMO_TASK_MAP; the last 8 human videos of each task are reserved for evaluation.
#
# Examples:
#   sbatch run_train_scripts/train_ur5e_vla_jepa_demo.sh                              # A, demo only
#   DEMO_ENCODER=qwen_video sbatch run_train_scripts/train_ur5e_vla_jepa_demo.sh      # C baseline
#   CONDITIONING=both LANGUAGE_DROPOUT=0.5 sbatch run_train_scripts/train_ur5e_vla_jepa_demo.sh

set -uo pipefail

# Hardcoded for the same reason as in train_ur5e_vla_jepa.sh (SLURM runs a spooled copy).
SCRIPT_PATH="/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/lerobot/lerobot/run_train_scripts/train_ur5e_vla_jepa_demo.sh"

source /hpc/apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate lerobot

CONDITIONING=${CONDITIONING:-demo}
DEMO_ENCODER=${DEMO_ENCODER:-vjepa}
LANGUAGE_DROPOUT=${LANGUAGE_DROPOUT:-0.0}
# Demo-text contrastive loss weight (option A); > 0 also logs demo_text_* similarity metrics.
CONTRASTIVE_WEIGHT=${CONTRASTIVE_WEIGHT:-0.0}
DEMO_TASK_FACTORS=${DEMO_TASK_FACTORS:-'{"color": 4, "bin": 4}'}
DEMO_ROOT=${DEMO_ROOT:-/mnt/beegfs/frosa/robot_datasets/dataset/opt_dataset/pick_place/human_rgb_pick_place/video}
DEMO_TASK_MAP=${DEMO_TASK_MAP:-/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/lerobot/lerobot/run_train_scripts/demo_task_maps/ur5e_pick_place_human.json}
DATASET_NAME=${DATASET_NAME:-ur5e_pick_place_delta_removed_0_5_10_15}
DATASET_REPO_ID=${DATASET_REPO_ID:-local/${DATASET_NAME}}
DATASET_ROOT=${DATASET_ROOT:-/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/open_x_embodiment/datasets/${DATASET_NAME}_lerobot}
PRETRAINED_PATH=${PRETRAINED_PATH:-lerobot/VLA-JEPA-Pretrain}
CON_TAG=$([ "${CONTRASTIVE_WEIGHT}" != "0.0" ] && [ "${CONTRASTIVE_WEIGHT}" != "0" ] && echo "_con${CONTRASTIVE_WEIGHT}")
JOB_NAME=${JOB_NAME:-ur5e_vla_jepa_${CONDITIONING}_${DEMO_ENCODER}${CON_TAG}_${DATASET_NAME}}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/${JOB_NAME}}
NUM_EPOCHS=${NUM_EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_PROCESSES=${NUM_PROCESSES:-4}
NUM_WORKERS=${NUM_WORKERS:-8}

export HF_HOME=${HF_HOME:-/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-"${HF_HOME}/hub"}
mkdir -p "${HUGGINGFACE_HUB_CACHE}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
# Unauthenticated Hub requests from 4 ranks at startup caused NCCL hangs in the pi0.5 runs.
export HF_HUB_OFFLINE=1

cd "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/lerobot/lerobot"

# Checkpoints at the END OF EVERY EPOCH (as in train_ur5e_pi05.sh): SAVE_FREQ = optimizer steps in one
# pass over the dataset = ceil(total_frames / (BATCH_SIZE * NUM_PROCESSES)), STEPS = NUM_EPOCHS * SAVE_FREQ,
# so checkpoints/<k*SAVE_FREQ> is the end of epoch k. After a chain resume the dataloader reshuffles,
# so "epoch k" means "k datasets' worth of samples seen".
TOTAL_FRAMES=$(python -c "import json; print(json.load(open('${DATASET_ROOT}/meta/info.json'))['total_frames'])")
SAVE_FREQ=$(( (TOTAL_FRAMES + BATCH_SIZE * NUM_PROCESSES - 1) / (BATCH_SIZE * NUM_PROCESSES) ))
STEPS=$(( NUM_EPOCHS * SAVE_FREQ ))
echo "Session ${JOB_NAME}: ${TOTAL_FRAMES} frames, effective batch $((BATCH_SIZE * NUM_PROCESSES)) -> ${SAVE_FREQ} steps/epoch, ${NUM_EPOCHS} epochs = ${STEPS} steps"

RESUME_CONFIG="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"

STEP_BEFORE=$(basename "$(readlink -f "${OUTPUT_DIR}/checkpoints/last" 2>/dev/null)" 2>/dev/null | sed 's/^0*//')
STEP_BEFORE=${STEP_BEFORE:-0}

if [ -f "${RESUME_CONFIG}" ]; then
  echo "Found existing checkpoint, resuming from ${RESUME_CONFIG} with steps=${STEPS}"
  TRAIN_ARGS=(--config_path="${RESUME_CONFIG}" --resume=true --steps="${STEPS}" --save_freq="${SAVE_FREQ}" --scheduler.num_decay_steps="${STEPS}" --wandb.disable_artifact=true)
else
  if [ -d "${OUTPUT_DIR}" ]; then
    echo "No checkpoint but ${OUTPUT_DIR} exists (stale dir from a crashed run) — clearing it before a fresh start"
    rm -rf "${OUTPUT_DIR}"
  fi
  echo "No existing checkpoint, starting fresh run (conditioning=${CONDITIONING}, demo_encoder=${DEMO_ENCODER})"
  TRAIN_ARGS=(
    --dataset.repo_id="${DATASET_REPO_ID}"
    --dataset.root="${DATASET_ROOT}"
    --dataset.video_backend=pyav
    --dataset.image_transforms.enable=true
    --policy.path="${PRETRAINED_PATH}"
    --policy.repo_id=local/${JOB_NAME}
    --policy.device=cuda
    --policy.reinit_modules='["model.action_model.action_encoder", "model.action_model.action_decoder", "model.action_model.state_encoder"]'
    --policy.push_to_hub=false
    --policy.conditioning="${CONDITIONING}"
    --policy.demo_encoder="${DEMO_ENCODER}"
    --policy.language_dropout="${LANGUAGE_DROPOUT}"
    --policy.demo_contrastive_weight="${CONTRASTIVE_WEIGHT}"
    --policy.demo_task_factors="${DEMO_TASK_FACTORS}"
    --policy.demo_root="${DEMO_ROOT}"
    --policy.demo_task_map_path="${DEMO_TASK_MAP}"
    --rename_map='{"observation.images.front":"observation.images.exterior_1_left","observation.images.gripper":"observation.images.exterior_2_left"}'
    --output_dir="${OUTPUT_DIR}"
    --job_name="${JOB_NAME}"
    --steps="${STEPS}"
    --policy.scheduler_decay_steps="${STEPS}"
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

# Self-chaining past gpuq's 7h walltime: see train_ur5e_vla_jepa.sh for the rationale.
# Env vars above are exported to the next link by `--export=ALL`, so it resumes the same run.
resubmit_if_progressed() {
  local step_after
  step_after=$(basename "$(readlink -f "${OUTPUT_DIR}/checkpoints/last" 2>/dev/null)" 2>/dev/null | sed 's/^0*//')
  step_after=${step_after:-0}
  if [ "${step_after}" -gt "${STEP_BEFORE}" ] && [ "${step_after}" -lt "${STEPS}" ]; then
    echo "Progressed ${STEP_BEFORE} -> ${step_after} (target ${STEPS} not yet reached) — submitting the next chain job"
    CONDITIONING="${CONDITIONING}" DEMO_ENCODER="${DEMO_ENCODER}" DATASET_NAME="${DATASET_NAME}" \
      CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT}" \
      JOB_NAME="${JOB_NAME}" OUTPUT_DIR="${OUTPUT_DIR}" NUM_EPOCHS="${NUM_EPOCHS}" \
      sbatch --job-name=vjepa_demo "${SCRIPT_PATH}" || echo "Self-resubmission failed (possibly at the cluster's MaxSubmit cap) — needs manual or external resubmission"
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

# Background srun + `wait` so the SIGTERM trap runs before SLURM's KillWait expires.
srun accelerate launch \
    --num_processes="${NUM_PROCESSES}" \
    --mixed_precision=bf16 \
    -m lerobot.scripts.lerobot_train \
    "${TRAIN_ARGS[@]}" &
wait $!
TRAIN_EXIT=$?

resubmit_if_progressed

exit "${TRAIN_EXIT}"
