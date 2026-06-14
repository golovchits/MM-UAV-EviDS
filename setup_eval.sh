#!/bin/bash
# Setup MM-UAV evaluation: copy GT + tracker results to eval toolkit structure
#
# Usage:
#   EXTERNAL=/path/to/MM-UAV-extracted/MMMUAV/test \
#   EVAL_SRC=/path/to/MM-UAV-Evaluation-ToolKit \
#   TRACKER_RESULTS=/path/to/YOLOX_outputs/<condition> \
#   bash setup_eval.sh

set -euo pipefail

EXTERNAL="${EXTERNAL:-/Volumes/My Passport/MM-UAV-extracted/MMMUAV/test}"
EVAL_SRC="${EVAL_SRC:-/path/to/MM-UAV-Evaluation-ToolKit}"
TRACKER_RESULTS="${TRACKER_RESULTS:-/path/to/YOLOX_outputs/<condition>}"

GT_BASE="${EVAL_SRC}/data/gt/MMMUAV"
TRACKER_BASE="${EVAL_SRC}/data/trackers/MMMUAV/baseline_defconv"

# ── 1. Create GT directories ────────────────────────────────────────────
echo "Setting up GT for RGB..."
for seq_dir in "${EXTERNAL}"/*/; do
    seq=$(basename "$seq_dir")
    gt_rgb="${seq_dir}/gt_rgb"
    if [ -d "$gt_rgb" ]; then
        mkdir -p "${GT_BASE}/MMMUAVrgb-test/${seq}/gt"
        cp "${gt_rgb}/gt.txt" "${GT_BASE}/MMMUAVrgb-test/${seq}/gt/"
    fi
done

echo "Setting up GT for IR..."
for seq_dir in "${EXTERNAL}"/*/; do
    seq=$(basename "$seq_dir")
    gt_ir="${seq_dir}/gt_ir"
    if [ -d "$gt_ir" ]; then
        mkdir -p "${GT_BASE}/MMMUAVir-test/${seq}/gt"
        cp "${gt_ir}/gt.txt" "${GT_BASE}/MMMUAVir-test/${seq}/gt/"
    fi
done

# ── 2. Create seqmap files ──────────────────────────────────────────────
mkdir -p "${GT_BASE}/seqmaps"
ls -d "${GT_BASE}/MMMUAVrgb-test"/*/ | xargs -n1 basename | sort > "${GT_BASE}/seqmaps/MMMUAVrgb-test.txt"
cp "${GT_BASE}/seqmaps/MMMUAVrgb-test.txt" "${GT_BASE}/seqmaps/MMMUAVir-test.txt"

echo "RGB sequences: $(wc -l < "${GT_BASE}/seqmaps/MMMUAVrgb-test.txt")"
echo "IR sequences:  $(wc -l < "${GT_BASE}/seqmaps/MMMUAVir-test.txt")"

# ── 3. Copy tracker results ─────────────────────────────────────────────
echo "Copying tracker results..."
mkdir -p "${TRACKER_BASE}"
cp -r "${TRACKER_RESULTS}/track_results_rgb" "${TRACKER_BASE}/"
cp -r "${TRACKER_RESULTS}/track_results_ir" "${TRACKER_BASE}/"

echo "RGB results: $(ls "${TRACKER_BASE}/track_results_rgb" | wc -l)"
echo "IR results:  $(ls "${TRACKER_BASE}/track_results_ir" | wc -l)"

echo "Done."
echo ""
echo "Now sync the eval toolkit to Snellius:"
echo "  rsync -avz \${EVAL_SRC}/ snellius:/path/to/code/MM-UAV-Evaluation-ToolKit/"
