#!/bin/bash
#SBATCH --job-name=e3_analysis_cde
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/e3_analysis_cde_%j.out
#SBATCH --error=logs/e3_analysis_cde_%j.err

CODE_DIR="${CODE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${CODE_DIR}"

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
source "${VENV_PATH:-/path/to/your/venv}/bin/activate"  # update VENV_PATH for your system
export PYTHONNOUSERSITE=1

echo "=== E3 Inference Analysis: conditions (c), (d), (e) — adding UAUC_vac ==="
echo "Started at $(date)"

python tools/e3_inference_analysis.py --cond c
echo "--- (c) done at $(date) ---"

python tools/e3_inference_analysis.py --cond d
echo "--- (d) done at $(date) ---"

python tools/e3_inference_analysis.py --cond e
echo "--- (e) done at $(date) ---"

echo "All done at $(date)"
echo "Output: tools/e3_analysis/e3_analysis_{c,d,e}.json (updated with uauc_vac)"
