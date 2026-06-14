#!/bin/bash
# =============================================================================
# E3 — Inference Analysis (ECE, UAUC, uncertainty histograms, agree-vs-disagree)
#
# Runs on val set for all conditions (a)-(e).
# Output: tools/e3_analysis/
#
# Submit with:
#   sbatch run_e3_analysis.slurm
# =============================================================================

#SBATCH --job-name=e3_analysis
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/e3_analysis_%j.out
#SBATCH --error=logs/e3_analysis_%j.err

set -euo pipefail

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
source "${VENV_PATH:-/path/to/your/venv}/bin/activate"  # update VENV_PATH for your system

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

CODE_DIR="${CODE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${CODE_DIR}"
mkdir -p logs tools/e3_analysis

echo "========================================"
echo "E3 Inference Analysis — All Conditions"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "========================================"

echo "Analysis start at $(date)"
python tools/e3_inference_analysis.py --all
echo "Analysis done at $(date)"

echo ""
echo "Output: tools/e3_analysis/e3_analysis_summary.json"
cat tools/e3_analysis/e3_analysis_summary.json | python -m json.tool | head -80
