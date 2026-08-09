#!/bin/bash
#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --exclude=gnode09

# Classic LIBERO evaluation with pi0.
# Runs the four standard suites (Spatial, Object, Goal, Long) at 10
# episodes/task, matching the protocol LeRobot uses for published results.
# See: docs/source/libero.mdx, docs/source/pi0.mdx

set -euo pipefail

source /hpc/apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate lerobot

export MUJOCO_GL=egl

POLICY_PATH=${POLICY_PATH:-lerobot/pi0_libero_finetuned_v044}
TASKS=${TASKS:-libero_spatial,libero_object,libero_goal,libero_10}
N_EPISODES=${N_EPISODES:-10}
BATCH_SIZE=${BATCH_SIZE:-1}
OUTPUT_DIR=${OUTPUT_DIR:-./eval_logs/pi0_libero}

export HF_HOME=${HF_HOME:-/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-"${HF_HOME}/hub"}
mkdir -p "${HUGGINGFACE_HUB_CACHE}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/.."

srun lerobot-eval \
    --policy.path="${POLICY_PATH}" \
    --env.type=libero \
    --env.task="${TASKS}" \
    --env.max_parallel_tasks=1 \
    --eval.batch_size="${BATCH_SIZE}" \
    --eval.n_episodes="${N_EPISODES}" \
    --output_dir="${OUTPUT_DIR}"
