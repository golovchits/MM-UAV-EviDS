#!/bin/bash
# =============================================================================
# E3 — ADFM confidence sweep: {0.3, 0.2, 0.1, 0.05} on 5 sequences
# Shows whether detection count recovers monotonically as conf drops.
# Submit: sbatch run_adfm_sweep.slurm
# =============================================================================

#SBATCH --job-name=adfm_sweep
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/adfm_sweep_%j.out
#SBATCH --error=logs/adfm_sweep_%j.err

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

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

BASE_OUT="YOLOX_outputs/yolox_s_2_evidential_adfm/sweep_results"
mkdir -p "${BASE_OUT}"

for conf in 0.3 0.2 0.1 0.05; do
    echo ""
    echo "========================================"
    echo "=== conf=${conf} ==="
    echo "Start: $(date)"

    python tools/masort-multi2-event.py \
        ${DATA_DIR:-/path/to/MM-UAV-images} \
        -f yolox/exps/example/custom/yolox_s_2_evidential_adfm.py \
        -c YOLOX_outputs/yolox_s_2_evidential_adfm/best_ckpt.pth.tar \
        --benchmark MMMUAV --fp16 --p 1 \
        --conf ${conf} \
        --track_high_thresh 0.1 --track_low_thresh 0.05 \
        --new_track_thresh 0.5 --appearance_thresh 0.20 \
        --proximity_thresh 0.9 --use_recent True \
        --with-reid --fuse --use_iou_3 --use_event_3 --use_app_3

    # Save before next run overwrites
    mkdir -p "${BASE_OUT}/conf_${conf}"
    cp -r YOLOX_outputs/yolox_s_2_evidential_adfm/track_results_rgb "${BASE_OUT}/conf_${conf}/"
    cp -r YOLOX_outputs/yolox_s_2_evidential_adfm/track_results_ir "${BASE_OUT}/conf_${conf}/"

    echo "Done: $(date)"
    echo "RGB detections:"
    for f in "${BASE_OUT}/conf_${conf}/track_results_rgb/"*.txt; do
        echo "  $(basename $f .txt): $(wc -l < $f)"
    done
done

echo ""
echo "========================================"
echo "=== Sweep Summary ==="
for conf in 0.3 0.2 0.1 0.05; do
    total=$(cat "${BASE_OUT}/conf_${conf}/track_results_rgb/"*.txt | wc -l)
    echo "conf=${conf}: ${total} total RGB detections across 5 seqs"
done
echo "========================================"
