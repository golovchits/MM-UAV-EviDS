#!/bin/bash
# =============================================================================
# E3 — Condition (c): Evidential + ADFM Stage 2 Training
#
# Evidential Dirichlet heads on both YOLOX streams.
# ADFM feature-level channel-attention fusion retained.
# Backbone + FPN frozen (except Fusion0/1/2 and heads).
#
# Submit with:
#   sbatch run_stage2_evidential_adfm.slurm
#
# SBU estimate: ~50 SBU (50 epochs, batch 16, A100)
# =============================================================================

#SBATCH --job-name=evi_adfm
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=15:00:00
#SBATCH --output=logs/stage2_evidential_adfm_%j.out
#SBATCH --error=logs/stage2_evidential_adfm_%j.err

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────
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
mkdir -p logs

echo "========================================"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "PyTorch      : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA         : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "========================================"

# ── Preload ALL training images into page cache ──────────────────────────────
echo "=== meminfo BEFORE warm-up ==="
grep -E "Cached:|MemAvailable:" /proc/meminfo

echo "Preloading all training images into page cache (this may take 5-15 min)..."
python -c "
import cv2, os, json
from multiprocessing.pool import ThreadPool
cv2.setNumThreads(0)

ann_dir = '${SCRATCH_DIR:-/path/to/scratch}/MM-UAV-tars/annotations'
img_dir = '${DATA_DIR:-/path/to/MM-UAV-images}'

# Collect unique file paths from train-rgb.json and train-ir.json
all_files = set()
for ann_name in ['train-rgb.json', 'train-ir.json']:
    with open(os.path.join(ann_dir, ann_name)) as f:
        data = json.load(f)
    for img in data['images']:
        # file_name matches exactly what training reads: data_dir + file_name
        # e.g. 'train/0002/rgb_frame/0001.jpg' -> MM-UAV-images/train/0002/...
        fn = img['file_name']
        all_files.add(os.path.join(img_dir, fn))
    print(f'  {ann_name}: {len(data[\"images\"])} images')

print(f'Total unique training images: {len(all_files)}')

paths = sorted(all_files)
count = [0]
bad = [0]
def read_one(p):
    img = cv2.imread(p)
    if img is None:
        bad[0] += 1
    count[0] += 1
    if count[0] % 5000 == 0:
        print(f'  {count[0]}/{len(paths)} cached, {bad[0]} failed so far...')
    return None

with ThreadPool(8) as pool:
    list(pool.imap(read_one, paths))
print(f'Done: {len(paths)} attempted, {bad[0]} FAILED (None from cv2.imread).')
if bad[0] > 0:
    print('WARNING: warm-up is reading wrong paths — page cache not filled!')
"

echo "=== meminfo AFTER warm-up ==="
grep -E "Cached:|MemAvailable:" /proc/meminfo
echo "Warm-up finished at $(date) (epoch seconds: $(date +%s))"

# ── Train ──────────────────────────────────────────────────────────────────
echo "Training start at $(date) (epoch seconds: $(date +%s))"
CUDA_VISIBLE_DEVICES=0 \
python -m yolox.train \
    -f yolox/exps/example/custom/yolox_s_2_evidential_adfm.py \
    --experiment-name yolox_s_2_evidential_adfm \
    -d 1 \
    -b 16 \
    --fp16

echo ""
echo "Job finished at $(date)"
echo "Checkpoint: YOLOX_outputs/yolox_s_2_evidential_adfm/best_ckpt.pth.tar"
