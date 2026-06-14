#!/bin/bash
#SBATCH --job-name=e3_seq_cis
#SBATCH --output=logs/e3_seq_cis_%x_%j.out
#SBATCH --error=logs/e3_seq_cis_%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu_a100

# Recompute E3 crop-control CIs with proper sequence-level clustering.
# Submit BOTH conditions in parallel from the shell:
#   sbatch --job-name=cis_d run_seq_cis.slurm d
#   sbatch --job-name=cis_e run_seq_cis.slurm e
#
# Each takes ~50 min on A100 (161K crops for d, 179K for e).

set -e
COND=${1:-d}

module purge && module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
source "${VENV_PATH:-/path/to/your/venv}/bin/activate"  # update VENV_PATH for your system

CODE_DIR="${CODE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${CODE_DIR}"

echo "=== Condition ($COND) ==="
python tools/recompute_sequence_cis.py --cond "$COND" --n-bootstrap 2000

echo ""
echo "Done. Results in ${SCRATCH_DIR:-/path/to/scratch}/e3_crop_control_v2/$COND/"
