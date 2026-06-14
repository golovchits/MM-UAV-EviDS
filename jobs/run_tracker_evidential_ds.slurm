#!/bin/bash
# =============================================================================
# E3 — Condition (e): Evidential + DS Fusion Tracker on ALL 121 test sequences
#
# ⚠️ Both streams return fused_output — IR tracking is the same coordinate-
# mismatch artifact. Report only RGB tracking metrics with †.
#
# Submit with:
#   sbatch run_tracker_evidential_ds.slurm
# =============================================================================

#SBATCH --job-name=track_ds
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/tracker_ds_%j.out
#SBATCH --error=logs/tracker_ds_%j.err

set -euo pipefail

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
source "${VENV_PATH:-/path/to/your/venv}/bin/activate"  # update VENV_PATH for your system
export PYTHONNOUSERSITE=1
CODE_DIR="${CODE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${CODE_DIR}"
mkdir -p logs

echo "========================================"
echo "E3 Condition (e): Evidential + DS Fusion"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "========================================"

echo "Pre-warming Lustre cache..."
python -c "
import cv2, os
cv2.setNumThreads(0)
data = '${DATA_DIR:-/path/to/MM-UAV-images}'
for d in sorted(os.listdir(data)):
    if not d.isdigit(): continue
    for m in ['rgb_frame', 'ir_frame', 'event_frame']:
        p = os.path.join(data, d, m)
        if os.path.isdir(p):
            _ = os.listdir(p)
for d in sorted(os.listdir(data))[:5]:
    if not d.isdigit(): continue
    for m in ['rgb_frame', 'ir_frame', 'event_frame']:
        p = os.path.join(data, d, m)
        if os.path.isdir(p):
            cv2.imread(os.path.join(p, sorted(os.listdir(p))[0]))
print('Done.')
"

python tools/masort-multi2-event.py \
    ${DATA_DIR:-/path/to/MM-UAV-images} \
    -f yolox/exps/example/custom/yolox_s_2_evidential_ds.py \
    -c YOLOX_outputs/yolox_s_2_evidential_ds/best_ckpt.pth.tar \
    --benchmark MMMUAV \
    --fp16 \
    --p 1 \
    --conf 0.3 \
    --track_high_thresh 0.1 \
    --track_low_thresh 0.1 \
    --new_track_thresh 0.7 \
    --appearance_thresh 0.20 \
    --proximity_thresh 0.9 \
    --use_recent True \
    --with-reid \
    --fuse \
    --use_iou_3 \
    --use_event_3 \
    --use_app_3

echo ""
echo "Job finished at $(date)"
echo "Results: YOLOX_outputs/yolox_s_2_evidential_ds/track_results_rgb/"
