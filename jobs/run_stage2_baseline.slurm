#!/bin/bash
# =============================================================================
# E3 — Stage 2 Baseline Reproduction (Condition a: MMA-SORT DefConv)
#
# Submit with:
#   sbatch run_stage2_baseline.slurm
#
# SBU estimate: ~50 SBU per run (50 epochs, batch 16, workers 4, individual files, FP16, A100)
# =============================================================================

#SBATCH --job-name=mmasort_stage2
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/stage2_baseline_%j.out
#SBATCH --error=logs/stage2_baseline_%j.err

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────
module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
source "${VENV_PATH:-/path/to/your/venv}/bin/activate"  # update VENV_PATH for your system

# Block ~/.local packages (PyTorch 2.11.0+cu130 conflicts with system CUDA 12.1.1)
export PYTHONNOUSERSITE=1
# Limit PyTorch internal thread pool to avoid futex deadlock with CUDA+OpenCV
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# ── Paths ──────────────────────────────────────────────────────────────────
CODE_DIR="${CODE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${CODE_DIR}"
mkdir -p logs

echo "========================================"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "GPU          : nvidia-smi"
echo "PyTorch      : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA         : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "========================================"

nvidia-smi

# ── Warm Lustre metadata cache ──────────────────────────────────────────────
echo "Pre-warming Lustre cache..."
python -c "
import cv2, os
cv2.setNumThreads(0)
data = '${DATA_DIR:-/path/to/MM-UAV-images}'
for d in sorted(os.listdir(data))[:20]:
    for m in ['rgb_frame', 'ir_frame']:
        p = os.path.join(data, d, m)
        if os.path.isdir(p):
            cv2.imread(os.path.join(p, sorted(os.listdir(p))[0]))
print('Done.')
"

# ── Train ──────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 \
python -m yolox.train \
    -f yolox/exps/example/custom/yolox_s_2_def-tuning-fusion-head.py \
    --experiment-name yolox_s_2_stream_def_tuning_fusion_head \
    -d 1 \
    -b 16 \
    --fp16

echo ""
echo "Job finished at $(date)"
echo "Checkpoint: YOLOX_outputs/yolox_s_2_stream_def_tuning_fusion_head/best_ckpt.pth.tar"
