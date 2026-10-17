#!/bin/bash
#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:50:00
#SBATCH --export=ALL
#SBATCH --exclude=gnode09

# LIBERO-plus evaluation of one shard (a contiguous range of task variants of one suite).
# LIBERO-plus has 10030 variants over the 4 standard suites (spatial 2402, object 2518,
# goal 2591, libero_10 2519); the benchmark protocol is 1 episode per variant, so the full run is
# split into shards (see run_libero_plus_all.sh) that each fit in one gpuq job.
#
# Usage: MODEL=<vla_jepa|pi05> SUITE=<libero_spatial|libero_object|libero_goal|libero_10> \
#        TASK_START=<int> TASK_END=<int, exclusive> sbatch run_libero_plus.sh
#
# LIBERO-plus is loaded WITHOUT touching the lerobot env (which keeps vanilla hf-libero for the
# classic LIBERO scripts): ../libero_plus_overlay/libero/__init__.py redirects `import libero` to the
# fork in repo/LIBERO-plus, ../libero_plus_overlay/site has scikit-image (needed by the fork), and
# ../libero_plus_overlay/config/config.yaml points LIBERO at the fork's assets/bddl/init files.

set -euo pipefail
source /hpc/apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate lerobot
PY=/mnt/beegfs/frosa/.conda/envs/lerobot/bin/python  # absolute: `python` can resolve to another env

MODEL=${MODEL:?MODEL required (vla_jepa | pi05)}
SUITE=${SUITE:?SUITE required}
TASK_START=${TASK_START:-0}
TASK_END=${TASK_END:-0}
[ -z "${TASK_IDS:-}" ] && [ "${TASK_END}" -le "${TASK_START}" ] && { echo "set TASK_START/TASK_END or TASK_IDS"; exit 1; }
N_EPISODES=${N_EPISODES:-1}

case "${MODEL}" in
  vla_jepa) POLICY_PATH=${POLICY_PATH:-lerobot/VLA-JEPA-LIBERO} ;;
  pi05)     POLICY_PATH=${POLICY_PATH:-lerobot/pi05_libero_finetuned_v044} ;;
  *) echo "unknown MODEL ${MODEL}"; exit 1 ;;
esac

REPO=/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/lerobot
OVERLAY=${REPO}/libero_plus_overlay
export PYTHONPATH=${OVERLAY}:${OVERLAY}/site${PYTHONPATH:+:${PYTHONPATH}}
export LIBERO_CONFIG_PATH=${OVERLAY}/config
export MUJOCO_GL=egl
export HF_HOME=${HF_HOME:-/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-"${HF_HOME}/hub"}
export HF_HUB_OFFLINE=1  # both checkpoints and their base models are in the local cache
# Videos: default 0 (one per variant would be ~10k mp4 per model plus an extra render per frame);
# set e.g. LEROBOT_EVAL_MAX_RENDERED=1 to keep one video per variant.
export LEROBOT_EVAL_MAX_RENDERED=${LEROBOT_EVAL_MAX_RENDERED:-0}
# Robot trajectories: one pickle per episode (eef pos/quat, gripper qpos, joints, actions, success, task)
# under <OUTPUT_DIR>/trajectories/<suite>_<task_id>/eval_episode_0.pkl, for gripper XY heat-maps
# (as LIBERO-plus/dataset_analysis.py). See _save_trajectories in lerobot_eval.py.
export LEROBOT_EVAL_SAVE_TRAJ=${LEROBOT_EVAL_SAVE_TRAJ:-1}

# TASK_IDS (optional, e.g. "[2091,2094]") overrides the contiguous [TASK_START, TASK_END) range.
TASK_IDS=${TASK_IDS:-"[$(seq -s, "${TASK_START}" $((TASK_END - 1)))]"}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO}/lerobot/eval_logs/libero_plus/${MODEL}/${SUITE}/$(printf '%04d_%04d' "${TASK_START}" "${TASK_END}")}
echo "MODEL=${MODEL} POLICY=${POLICY_PATH} SUITE=${SUITE} task_ids ${TASK_IDS} -> ${OUTPUT_DIR}"

cd ${REPO}/lerobot
srun ${PY} -m lerobot.scripts.lerobot_eval \
    --policy.path="${POLICY_PATH}" \
    --env.type=libero_plus \
    --env.task="${SUITE}" \
    --env.task_ids="${TASK_IDS}" \
    --env.max_parallel_tasks=1 \
    --eval.batch_size=1 \
    --eval.n_episodes="${N_EPISODES}" \
    --output_dir="${OUTPUT_DIR}"
