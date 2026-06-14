#!/bin/bash
# =============================================================================
# E3 — Tracker Evaluation: compute MOTA/HOTA/IDF1 for conditions (c)-(e)
#
# Copies tracker outputs to eval toolkit and runs TrackEval.
#
# Submit with:
#   sbatch run_e3_trackeval.slurm
# =============================================================================

#SBATCH --job-name=trackeval
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/trackeval_%j.out
#SBATCH --error=logs/trackeval_%j.err

set -euo pipefail

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
source "${VENV_PATH:-/path/to/your/venv}/bin/activate"  # update VENV_PATH for your system

CODE_DIR="${CODE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
EVAL_DIR="${EVAL_DIR:-/path/to/MM-UAV-Evaluation-ToolKit}"  # update for your system
DATA_DIR="${DATA_DIR:-/path/to/MM-UAV-images}"

cd "${CODE_DIR}"
mkdir -p logs

echo "========================================"
echo "E3 Tracker Evaluation"
echo "Job ID : ${SLURM_JOB_ID}"
echo "========================================"

# ── 1. Set up GT (from extracted dataset test sequences) ──────────────────
echo "Setting up GT..."
GT_BASE="${EVAL_DIR}/data/gt/MMMUAV"

# Only regenerate GT if not already present (baseline_defconv already set it up)
SEQMAP="${GT_BASE}/seqmaps/MMMUAVrgb-test.txt"
if [ -f "${SEQMAP}" ]; then
    echo "GT already set up: $(wc -l < "${SEQMAP}") sequences — skipping GT copy"
else
    echo "Setting up GT from test sequences only..."
    # Use DATA_DIR/test/ — root contains BOTH train+test, test/ is test-only
    for seq_dir in "${DATA_DIR}"/test/*/; do
        seq=$(basename "$seq_dir")
        if [ -d "${seq_dir}/gt_rgb" ]; then
            mkdir -p "${GT_BASE}/MMMUAVrgb-test/${seq}/gt"
            cp "${seq_dir}/gt_rgb/gt.txt" "${GT_BASE}/MMMUAVrgb-test/${seq}/gt/"
        fi
        if [ -d "${seq_dir}/gt_ir" ]; then
            mkdir -p "${GT_BASE}/MMMUAVir-test/${seq}/gt"
            cp "${seq_dir}/gt_ir/gt.txt" "${GT_BASE}/MMMUAVir-test/${seq}/gt/"
        fi
    done
    mkdir -p "${GT_BASE}/seqmaps"
    ls -d "${GT_BASE}/MMMUAVrgb-test"/*/ | xargs -n1 basename | sort > "${SEQMAP}"
    cp "${SEQMAP}" "${GT_BASE}/seqmaps/MMMUAVir-test.txt"
    echo "GT setup done: $(wc -l < "${SEQMAP}") sequences"
fi

# ── 2. Copy tracker results ──────────────────────────────────────────────
echo "Copying tracker results..."

# Only copy new conditions — baselines (a/b) already in eval toolkit with results
# tracker_fol = TRACKERS_FOLDER (split NOT added, per mot_challenge_2d_box.py:58)
for cond in evidential_adfm evidential_average evidential_ds; do
    SRC="${CODE_DIR}/YOLOX_outputs/yolox_s_2_${cond}"
    DST="${EVAL_DIR}/data/trackers/MMMUAV/${cond}"
    mkdir -p "${DST}"
    cp -r "${SRC}/track_results_rgb" "${DST}/"
    cp -r "${SRC}/track_results_ir" "${DST}/"
    echo "  ${cond}: $(ls "${DST}/track_results_rgb/" | wc -l) RGB files"
done

# ── 3. Run evaluation ────────────────────────────────────────────────────
cd "${EVAL_DIR}"

echo ""
echo "========== Seqmap check (must be 121) =========="
wc -l "${GT_BASE}/seqmaps/MMMUAVrgb-test.txt"

echo ""
echo "========== RGB Evaluation =========="
python scripts/run_mot_challenge_2modalities.py \
  --BENCHMARK MMMUAVrgb \
  --TRACKERS_TO_EVAL evidential_adfm evidential_average evidential_ds \
  --SPLIT_TO_EVAL test \
  --TRACKER_SUB_FOLDER track_results_rgb \
  --OUTPUT_SUB_FOLDER eval_results_rgb

echo ""
echo "========== IR Evaluation =========="
python scripts/run_mot_challenge_2modalities.py \
  --BENCHMARK MMMUAVir \
  --TRACKERS_TO_EVAL evidential_adfm evidential_average evidential_ds \
  --SPLIT_TO_EVAL test \
  --TRACKER_SUB_FOLDER track_results_ir \
  --OUTPUT_SUB_FOLDER eval_results_ir

echo ""
echo "Done at $(date)"
echo "Results in ${EVAL_DIR}/data/trackers/MMMUAV/{tracker}/eval_results_rgb/"
