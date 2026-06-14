#!/bin/bash
# =============================================================================
# E3 — Condition (g): Baseline + Temperature Scaling
# Step 1: Fit T on 120 held-out train seqs (~4h)
# Step 2: Run tracker (auto-detects temperature.json) on 121 test seqs (~2h)
# Step 3: Copy + eval (~30min)
# Submit: sbatch run_temp_scale_full.slurm
# =============================================================================

#SBATCH --job-name=temp_scale
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/temp_scale_%j.out
#SBATCH --error=logs/temp_scale_%j.err

set -euo pipefail

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
source "${VENV_PATH:-/path/to/your/venv}/bin/activate"  # update VENV_PATH for your system
export PYTHONNOUSERSITE=1

CODE_DIR="${CODE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
EVAL_DIR="${EVAL_DIR:-/path/to/MM-UAV-Evaluation-ToolKit}"  # update for your system

cd "${CODE_DIR}"
mkdir -p logs

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# ── Step 1: Fit temperature ───────────────────────────────────────────────
echo "=== Step 1: Fitting temperature on tune split ==="
python tools/fit_temperature.py

T_VAL=$(python -c "import json; print(json.load(open('YOLOX_outputs/yolox_s_2_def-tuning-fusion-head/temperature.json'))['T'])")
echo "Fitted T = ${T_VAL}"

# ── Step 2: Run tracker (auto-detects temperature.json) ───────────────────
echo ""
echo "=== Step 2: Tracker on 121 test sequences ==="
echo "Start: $(date)"

python tools/masort-multi2-event.py \
    ${DATA_DIR:-/path/to/MM-UAV-images} \
    -f yolox/exps/example/custom/yolox_s_2_def-tuning-fusion-head.py \
    -c YOLOX_outputs/yolox_s_2_def-tuning-fusion-head/best_ckpt.pth.tar \
    --benchmark MMMUAV --fp16 --p 1 \
    --conf 0.3 \
    --track_high_thresh 0.1 --track_low_thresh 0.1 \
    --new_track_thresh 0.7 --appearance_thresh 0.20 \
    --proximity_thresh 0.9 --use_recent True \
    --with-reid --fuse --use_iou_3 --use_event_3 --use_app_3

echo "Tracker done: $(date)"

# ── Step 3: Copy to eval toolkit ──────────────────────────────────────────
TRACKER_NAME="baseline_temp_scale"
DST="${EVAL_DIR}/data/trackers/MMMUAV/${TRACKER_NAME}"
mkdir -p "${DST}"
cp -r YOLOX_outputs/yolox_s_2_def-tuning-fusion-head/track_results_rgb "${DST}/"
cp -r YOLOX_outputs/yolox_s_2_def-tuning-fusion-head/track_results_ir "${DST}/"
echo "Copied: $(ls "${DST}/track_results_rgb/" | wc -l) files"

# ── Step 4: Eval ──────────────────────────────────────────────────────────
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
echo "Done at $(date)"
