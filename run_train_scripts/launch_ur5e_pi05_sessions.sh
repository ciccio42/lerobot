#!/bin/bash
# Submits every pi0.5 UR5e training session — the same sessions trained for VLA-JEPA
# (outputs/ur5e_vla_jepa*): six sim task splits from lerobot/pi05_base, plus the real-robot
# session warm-started from the finished sim 0/5/10/15 checkpoint (submitted automatically by
# that sim chain's last job via FOLLOWUP_SCRIPT, like train_ur5e_vla_jepa_real.sh's recipe).
#
# Usage (from anywhere, on the login node):
#   bash run_train_scripts/launch_ur5e_pi05_sessions.sh            # all sessions
#   bash run_train_scripts/launch_ur5e_pi05_sessions.sh rm_one_spawn delta_all   # a subset
# Extra env vars (NUM_EPOCHS, BATCH_SIZE, WANDB_PROJECT, ...) are forwarded to every session.

set -euo pipefail

LEROBOT_DIR=/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/lerobot/lerobot
DATA_DIR=/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/open_x_embodiment/datasets
TRAIN_SCRIPT=${LEROBOT_DIR}/run_train_scripts/train_ur5e_pi05.sh
REAL_FOLLOWUP=${LEROBOT_DIR}/run_train_scripts/launch_ur5e_pi05_real.sh

# session key -> dataset name (dataset dir is ${DATA_DIR}/<name>_lerobot)
declare -A SESSIONS=(
  [delta_all]=ur5e_pick_place_delta_all
  [removed_spawn_regions]=ur5e_pick_place_removed_spawn_regions
  [rm_central_spawn]=ur5e_pick_place_rm_central_spawn
  [rm_one_spawn]=ur5e_pick_place_rm_one_spawn
  [delta_removed_0_5_10_15]=ur5e_pick_place_delta_removed_0_5_10_15
  [rm_12_13_14_15]=ur5e_pick_place_rm_12_13_14_15
)

SELECTED=("$@")
[ ${#SELECTED[@]} -eq 0 ] && SELECTED=("${!SESSIONS[@]}")

cd "${LEROBOT_DIR}"
for key in "${SELECTED[@]}"; do
  if [ "${key}" = "real" ]; then
    bash "${REAL_FOLLOWUP}"
    continue
  fi
  ds=${SESSIONS[$key]:?unknown session '${key}' (valid: ${!SESSIONS[*]} real)}
  job=ur5e_pi05_${key}
  followup=""
  [ "${key}" = "delta_removed_0_5_10_15" ] && followup=${REAL_FOLLOWUP}
  echo "Submitting ${job} (${ds})"
  DATASET_REPO_ID=local/${ds} \
  DATASET_ROOT=${DATA_DIR}/${ds}_lerobot \
  PRETRAINED_PATH=lerobot/pi05_base \
  JOB_NAME=${job} \
  OUTPUT_DIR=outputs/${job} \
  FOLLOWUP_SCRIPT=${followup} \
    sbatch --job-name="${job}" "${TRAIN_SCRIPT}"
done
