#!/bin/bash
#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL

# Classic LIBERO evaluation with pi0fast.
# Runs the four standard suites (Spatial, Object, Goal, Long) at 10
# episodes/task, matching the protocol LeRobot uses for published results.
# pi0fast-libero expects observation.images.{base_0_rgb,left_wrist_0_rgb},
# so the LIBERO env keys are remapped via --rename_map.
# See: docs/source/libero.mdx, docs/source/pi0fast.mdx, docs/source/rename_map.mdx

set -euo pipefail

export MUJOCO_GL=egl

POLICY_PATH=${POLICY_PATH:-lerobot/pi0fast-libero}
TASKS=${TASKS:-libero_spatial,libero_object,libero_goal,libero_10}
N_EPISODES=${N_EPISODES:-10}
BATCH_SIZE=${BATCH_SIZE:-1}
OUTPUT_DIR=${OUTPUT_DIR:-./eval_logs/pi0fast_libero}

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/.."

srun lerobot-eval \
    --policy.path="${POLICY_PATH}" \
    --policy.max_action_tokens=256 \
    --policy.gradient_checkpointing=false \
    --env.type=libero \
    --env.task="${TASKS}" \
    --env.max_parallel_tasks=1 \
    --eval.batch_size="${BATCH_SIZE}" \
    --eval.n_episodes="${N_EPISODES}" \
    --output_dir="${OUTPUT_DIR}" \
    --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
