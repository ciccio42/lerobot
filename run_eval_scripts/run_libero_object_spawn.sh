#!/bin/bash
#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL

# Customized libero_object evaluation with spawn-region randomization.
#
# Ports the openvla-oft `run_libero_eval.sh` / `run_libero_eval.py` behavior
# (--change_spawn True --spawn_train_distribution False) to LeRobot's own
# LIBERO env, which implements the same mechanism natively: every episode
# rebuilds the MuJoCo scene from a bddl file with a freshly-sampled
# floor_target_object_region for the target object. Only libero_object ships
# a spawn_region.json, so this only makes sense for that suite.
# See: docs/source/libero.mdx#spawn-region-randomization-libero_object
#
# Usage: sbatch run_libero_object_spawn.sh [pi0|pi0fast]  (default: pi0)

set -euo pipefail

export MUJOCO_GL=egl

POLICY_TYPE=${1:-pi0}
SPAWN_TRAIN_DISTRIBUTION=${SPAWN_TRAIN_DISTRIBUTION:-false}
N_EPISODES=${N_EPISODES:-10}
BATCH_SIZE=${BATCH_SIZE:-2}

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/.."

case "${POLICY_TYPE}" in
  pi0)
    POLICY_PATH=${POLICY_PATH:-lerobot/pi0_libero}
    EXTRA_ARGS=()
    ;;
  pi0fast)
    POLICY_PATH=${POLICY_PATH:-lerobot/pi0fast-libero}
    EXTRA_ARGS=(
      --policy.max_action_tokens=256
      --policy.gradient_checkpointing=false
      --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
    )
    ;;
  *)
    echo "Unknown policy type '${POLICY_TYPE}', expected 'pi0' or 'pi0fast'" >&2
    exit 1
    ;;
esac

OUTPUT_DIR=${OUTPUT_DIR:-./eval_logs/libero_object_spawn_${POLICY_TYPE}}

srun lerobot-eval \
    --policy.path="${POLICY_PATH}" \
    --env.type=libero \
    --env.task=libero_object \
    --env.change_spawn=true \
    --env.spawn_train_distribution="${SPAWN_TRAIN_DISTRIBUTION}" \
    --eval.batch_size="${BATCH_SIZE}" \
    --eval.n_episodes="${N_EPISODES}" \
    --output_dir="${OUTPUT_DIR}" \
    "${EXTRA_ARGS[@]}"
