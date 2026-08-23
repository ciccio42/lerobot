#!/bin/bash
#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --exclude=gnode09

# Classic LIBERO evaluation with MolmoAct2 (LeRobot-saved checkpoint).
# Runs the four standard suites (Spatial, Object, Goal, Long), matching the
# protocol in docs/source/molmoact2.mdx's "Evaluation With LeRobot MolmoAct2
# Weight" section.
# See: docs/source/libero.mdx, docs/source/molmoact2.mdx

set -euo pipefail

source /hpc/apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate lerobot

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

POLICY_PATH=${POLICY_PATH:-allenai/MolmoAct2-LIBERO-LeRobot}
TASKS=${TASKS:-libero_spatial,libero_object,libero_goal,libero_10}
N_EPISODES=${N_EPISODES:-10}
BATCH_SIZE=${BATCH_SIZE:-1}
OUTPUT_DIR=${OUTPUT_DIR:-./eval_logs/molmoact2}

export HF_HOME=${HF_HOME:-/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-"${HF_HOME}/hub"}
mkdir -p "${HUGGINGFACE_HUB_CACHE}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/.."

srun lerobot-eval \
    --policy.path="${POLICY_PATH}" \
    --policy.inference_action_mode=continuous \
    --policy.model_dtype=bfloat16 \
    --policy.use_amp=true \
    --policy.enable_inference_cuda_graph=true \
    --policy.device=cuda \
    --policy.per_episode_seed=true \
    --policy.eval_seed=1000 \
    --env.type=libero \
    --env.task="${TASKS}" \
    --env.max_parallel_tasks=1 \
    --env.camera_name_mapping='{"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}' \
    --eval.batch_size="${BATCH_SIZE}" \
    --eval.n_episodes="${N_EPISODES}" \
    --output_dir="${OUTPUT_DIR}" \
    --seed=1000
