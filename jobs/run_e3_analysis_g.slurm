#!/bin/bash
#SBATCH --job-name=e3_analysis_g
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/e3_analysis_g_%j.out
#SBATCH --error=logs/e3_analysis_g_%j.err

CODE_DIR="${CODE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${CODE_DIR}"

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
source "${VENV_PATH:-/path/to/your/venv}/bin/activate"  # update VENV_PATH for your system
export PYTHONNOUSERSITE=1

echo "=== E3 Analysis: condition (g) Temperature Scaling ==="
echo "Requires: YOLOX_outputs/yolox_s_2_def-tuning-fusion-head/temperature.json"
echo "Started at $(date)"

# Verify temperature.json exists before starting the long analysis
if [ ! -f YOLOX_outputs/yolox_s_2_def-tuning-fusion-head/temperature.json ]; then
    echo "ERROR: temperature.json not found. Run fit_temperature.py first."
    exit 1
fi

echo "Using T=$(python3 -c "import json; d=json.load(open('YOLOX_outputs/yolox_s_2_def-tuning-fusion-head/temperature.json')); print(f\"{d['T']:.4f}\")")"

python tools/e3_inference_analysis.py --cond g --output-dir tools/e3_analysis

echo "Done at $(date)"
echo ""
echo "Output: tools/e3_analysis/e3_analysis_g.json"
echo "Compare with (a) baseline to see ECE improvement direction."
