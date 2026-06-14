#!/bin/bash
#SBATCH --job-name=e3_crop_extract
#SBATCH --output=logs/e3_crop_extract_%j.out
#SBATCH --error=logs/e3_crop_extract_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# ==========================================================================
# E3 Balanced-Crop Positive Control — Step 1: Extract Detection Crops
# ==========================================================================
# Runs the EDL detection pipeline on the MM-UAV test set and extracts
# image crops around every post-NMS detection with TP/FP labels.
#
# Expected runtime: ~3-4 hours (121 test seqs × ~13K images on A100)
# Output: tools/e3_crops/{cond}/metadata.json + crops/
# ==========================================================================

set -e
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

# ── Environment ──────────────────────────────────────────────────────────
module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
source "${VENV_PATH:-/path/to/your/venv}/bin/activate"  # update VENV_PATH for your system

export CUDA_HOME=/sw/arch/RHEL8/EB_production/2023/software/CUDA/12.1.1
export TORCH_CUDA_ARCH_LIST="8.0"

# ── Paths ────────────────────────────────────────────────────────────────
CODE_DIR="${CODE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd $CODE_DIR

mkdir -p logs

# ── Run ──────────────────────────────────────────────────────────────────
CROP_DIR=${SCRATCH_DIR:-/path/to/scratch}/e3_crops
mkdir -p "$CROP_DIR"

echo "Extracting detection crops for condition (d) — Evidential + Average..."
python tools/extract_detection_crops.py --cond d --out-dir "$CROP_DIR"

echo ""
echo "Extracting detection crops for condition (e) — Evidential + DS..."
python tools/extract_detection_crops.py --cond e --out-dir "$CROP_DIR"

echo ""
echo "Step 1 complete at $(date)"
echo "Next: submit run_crop_inference.slurm"
