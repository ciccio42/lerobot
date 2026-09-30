#!/bin/bash
# Submits the real-robot pi0.5 session: fine-tunes on real_ur5e_pick_place_delta_removed_0_5_10_15
# (480 episodes / 25704 frames, same 12 train tasks + held-out 0/5/10/15 split and same feature
# schema as the sim dataset), warm-started from the FINAL checkpoint of the sim
# ur5e_pi05_delta_removed_0_5_10_15 session — the pi0.5 mirror of train_ur5e_vla_jepa_real.sh.
#
# Normally run automatically by the sim session's last chain job (FOLLOWUP_SCRIPT); can also be
# run by hand once that session has finished.

set -euo pipefail

LEROBOT_DIR=/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/lerobot/lerobot
SIM_LAST=${LEROBOT_DIR}/outputs/ur5e_pi05_delta_removed_0_5_10_15/checkpoints/last
# Resolve the symlink now so the warm start is pinned to a concrete step directory.
SIM_CKPT=$(readlink -f "${SIM_LAST}")/pretrained_model
[ -f "${SIM_CKPT}/config.json" ] || { echo "Sim warm-start checkpoint not found at ${SIM_CKPT}"; exit 1; }

cd "${LEROBOT_DIR}"
echo "Submitting ur5e_pi05_real (warm start: ${SIM_CKPT})"
DATASET_REPO_ID=local/real_ur5e_pick_place_delta_removed_0_5_10_15 \
DATASET_ROOT=/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/open_x_embodiment/datasets/real_ur5e_pick_place_delta_removed_0_5_10_15_lerobot \
PRETRAINED_PATH=${SIM_CKPT} \
JOB_NAME=ur5e_pi05_real \
OUTPUT_DIR=outputs/ur5e_pi05_real \
FOLLOWUP_SCRIPT= \
  sbatch --job-name=ur5e_pi05_real "${LEROBOT_DIR}/run_train_scripts/train_ur5e_pi05.sh"
