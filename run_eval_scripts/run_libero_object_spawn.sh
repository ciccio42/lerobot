#!/bin/bash
#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --exclude=gnode09

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
# Usage: sbatch run_libero_object_spawn.sh [pi0|pi0fast|pi05|molmoact2|vla_jepa]
#   (default: pi0)

set -euo pipefail

source /hpc/apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate lerobot

export MUJOCO_GL=egl

POLICY_TYPE=${1:-pi0}
SPAWN_TRAIN_DISTRIBUTION=${SPAWN_TRAIN_DISTRIBUTION:-false}
N_EPISODES=${N_EPISODES:-10}

export HF_HOME=${HF_HOME:-/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-"${HF_HOME}/hub"}
mkdir -p "${HUGGINGFACE_HUB_CACHE}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/.."

case "${POLICY_TYPE}" in
  pi0)
    POLICY_PATH=${POLICY_PATH:-lerobot/pi0_libero}
    BATCH_SIZE_DEFAULT=2
    EXTRA_ARGS=()
    ;;
  pi0fast)
    POLICY_PATH=${POLICY_PATH:-lerobot/pi0fast-libero}
    BATCH_SIZE_DEFAULT=2
    EXTRA_ARGS=(
      --policy.max_action_tokens=256
      --policy.gradient_checkpointing=false
      --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
    )
    ;;
  pi05)
    POLICY_PATH=${POLICY_PATH:-lerobot/pi05_libero_finetuned_v044}
    BATCH_SIZE_DEFAULT=2
    EXTRA_ARGS=()
    ;;
  molmoact2)
    POLICY_PATH=${POLICY_PATH:-allenai/MolmoAct2-LIBERO-LeRobot}
    BATCH_SIZE_DEFAULT=1
    export PYOPENGL_PLATFORM=egl
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    EXTRA_ARGS=(
      --policy.inference_action_mode=continuous
      --policy.model_dtype=bfloat16
      --policy.use_amp=true
      --policy.enable_inference_cuda_graph=true
      --policy.device=cuda
      --policy.per_episode_seed=true
      --policy.eval_seed=1000
      --env.camera_name_mapping='{"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}'
      --seed=1000
    )
    ;;
  vla_jepa)
    POLICY_PATH=${POLICY_PATH:-lerobot/VLA-JEPA-LIBERO}
    BATCH_SIZE_DEFAULT=5
    EXTRA_ARGS=()
    ;;
  *)
    echo "Unknown policy type '${POLICY_TYPE}', expected 'pi0', 'pi0fast', 'pi05', 'molmoact2' or 'vla_jepa'" >&2
    exit 1
    ;;
esac

BATCH_SIZE=${BATCH_SIZE:-$BATCH_SIZE_DEFAULT}
DIST_SUFFIX=$([ "${SPAWN_TRAIN_DISTRIBUTION}" = "true" ] && echo "train_dist" || echo "ood_dist")
OUTPUT_DIR=${OUTPUT_DIR:-./eval_logs/libero_object_spawn_${POLICY_TYPE}_${DIST_SUFFIX}}

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
