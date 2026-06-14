#!/bin/bash
#SBATCH --job-name=e3_finetune
#SBATCH --output=logs/e3_finetune_%j.out
#SBATCH --error=logs/e3_finetune_%j.err
#SBATCH --time=05:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# ==========================================================================
# E3 Balanced-Crop Positive Control — Step 2 (Redesigned): Fine-Tune Head
# ==========================================================================
# Fine-tunes the evidential classification head on balanced detection crops,
# then measures vacuity UAUC with sequence-clustered bootstrap CI.
#
# Depends on Step 1 output: ${SCRATCH_DIR:-/path/to/scratch}/e3_crops/{cond}/
#
# Expected runtime: ~15-30 min on A100
# ==========================================================================

set -e
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"

module purge && module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
source "${VENV_PATH:-/path/to/your/venv}/bin/activate"  # update VENV_PATH for your system
export CUDA_HOME=/sw/arch/RHEL8/EB_production/2023/software/CUDA/12.1.1

CROP_DIR=${SCRATCH_DIR:-/path/to/scratch}/e3_crops
OUT_DIR=${SCRATCH_DIR:-/path/to/scratch}/e3_finetune
CODE_DIR="${CODE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd $CODE_DIR
mkdir -p logs

# Verify Step 1 output exists
for cond in d e; do
    if [ ! -f "${CROP_DIR}/${cond}/metadata.json" ]; then
        echo "ERROR: ${CROP_DIR}/${cond}/metadata.json not found."
        echo "Run extraction step first."
        exit 1
    fi
    N=$(python -c "import json; d=json.load(open('${CROP_DIR}/${cond}/metadata.json')); print(d.get('n_crops_saved', 0))")
    echo "Condition ($cond): $N crops available"
done

# Fine-tune condition (d)
echo ""
echo "=== Fine-tuning condition (d) ==="
python tools/finetune_crops_edl.py \
    --cond d --crop-dir "$CROP_DIR" --out-dir "$OUT_DIR" \
    --epochs 10 --lr 1e-3 --batch-size 32

# Fine-tune condition (e)
echo ""
echo "=== Fine-tuning condition (e) ==="
python tools/finetune_crops_edl.py \
    --cond e --crop-dir "$CROP_DIR" --out-dir "$OUT_DIR" \
    --epochs 10 --lr 1e-3 --batch-size 32

echo ""
echo "Done at $(date)"
echo "Results: $OUT_DIR/{d,e}/finetune_results.json"
