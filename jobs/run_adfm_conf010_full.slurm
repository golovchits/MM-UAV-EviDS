#!/bin/bash
# =============================================================================
# E3 — ADFM full run with conf=0.1 + eval
# Submit: sbatch run_adfm_conf010_full.slurm
# =============================================================================

#SBATCH --job-name=adfm_010_full
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/adfm_010_full_%j.out
#SBATCH --error=logs/adfm_010_full_%j.err

set -euo pipefail

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
source "${VENV_PATH:-/path/to/your/venv}/bin/activate"  # update VENV_PATH for your system
export PYTHONNOUSERSITE=1

CODE_DIR="${CODE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
EVAL_DIR="${EVAL_DIR:-/path/to/MM-UAV-Evaluation-ToolKit}"  # update for your system
DATA_DIR="${DATA_DIR:-/path/to/MM-UAV-images}"

cd "${CODE_DIR}"
mkdir -p logs

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# ── Warm Lustre cache ─────────────────────────────────────────────────────
echo "Pre-warming Lustre cache..."
python -c "
import cv2, os
cv2.setNumThreads(0)
data = '${DATA_DIR}'
for d in sorted(os.listdir(data)):
    if not d.isdigit(): continue
    for m in ['rgb_frame', 'ir_frame', 'event_frame']:
        p = os.path.join(data, d, m)
        if os.path.isdir(p):
            _ = os.listdir(p)
print('Done.')
"

# ── Tracker ───────────────────────────────────────────────────────────────
echo "Start: $(date)"
python tools/masort-multi2-event.py \
    ${DATA_DIR:-/path/to/MM-UAV-images} \
    -f yolox/exps/example/custom/yolox_s_2_evidential_adfm.py \
    -c YOLOX_outputs/yolox_s_2_evidential_adfm/best_ckpt.pth.tar \
    --benchmark MMMUAV --fp16 --p 1 \
    --conf 0.1 \
    --track_high_thresh 0.1 --track_low_thresh 0.1 \
    --new_track_thresh 0.5 --appearance_thresh 0.20 \
    --proximity_thresh 0.9 --use_recent True \
    --with-reid --fuse --use_iou_3 --use_event_3 --use_app_3

echo "Tracker done: $(date)"

# ── Copy to eval toolkit ──────────────────────────────────────────────────
TRACKER_NAME="evidential_adfm_conf010"
DST="${EVAL_DIR}/data/trackers/MMMUAV/${TRACKER_NAME}"
mkdir -p "${DST}"
cp -r YOLOX_outputs/yolox_s_2_evidential_adfm/track_results_rgb "${DST}/"
cp -r YOLOX_outputs/yolox_s_2_evidential_adfm/track_results_ir "${DST}/"
echo "Copied: $(ls "${DST}/track_results_rgb/" | wc -l) RGB, $(ls "${DST}/track_results_ir/" | wc -l) IR files"

# ── Eval ──────────────────────────────────────────────────────────────────
cd "${EVAL_DIR}"

echo "=== RGB Eval ==="
python scripts/run_mot_challenge_2modalities.py \
  --BENCHMARK MMMUAVrgb \
  --TRACKERS_TO_EVAL "${TRACKER_NAME}" \
  --SPLIT_TO_EVAL test \
  --TRACKER_SUB_FOLDER track_results_rgb \
  --OUTPUT_SUB_FOLDER eval_results_rgb

echo "=== IR Eval ==="
python scripts/run_mot_challenge_2modalities.py \
  --BENCHMARK MMMUAVir \
  --TRACKERS_TO_EVAL "${TRACKER_NAME}" \
  --SPLIT_TO_EVAL test \
  --TRACKER_SUB_FOLDER track_results_ir \
  --OUTPUT_SUB_FOLDER eval_results_ir

echo ""
echo "=== RGB Summary ==="
cat "${DST}/eval_results_rgb/drone_summary_light.txt" 2>/dev/null || echo "(missing)"

echo ""
echo "=== Decision: MOTA >30 = threshold confound. MOTA <10 = structural. ==="
echo "Done at $(date)"
