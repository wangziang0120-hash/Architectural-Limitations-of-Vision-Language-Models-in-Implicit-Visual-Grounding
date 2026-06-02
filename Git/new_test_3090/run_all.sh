#!/bin/bash
# Run all evaluations and report generation
# 3 Instruct models × 5 methods = 15 experiments

set -e

echo "=========================================="
echo "VLM Grounding Evaluation - Instruct Models"
echo "=========================================="

# Uncomment to use HuggingFace mirror (for users in China)
# export HF_ENDPOINT=https://hf-mirror.com

echo "[0] Installing dependencies..."
pip install transformers torch pillow opencv-python-headless numpy openpyxl accelerate -q

MODELS=("8b_inst" "4b_inst" "2b_inst")
METHODS=("baseline" "prompt" "blur" "blackout" "crop")

step=1
total=$((${#MODELS[@]} * (${#METHODS[@]} + 2)))

for model in "${MODELS[@]}"; do
    echo ""
    echo "===== Testing model: $model ====="

    for method in "${METHODS[@]}"; do
        echo "[${step}/${total}] $model - $method"
        python eval.py "$model" "$method"
        ((step++))
    done

    echo "[${step}/${total}] Generating Excel: $model"
    python generate_excel.py "$model"
    ((step++))

    echo "[${step}/${total}] Generating HTML report: $model"
    python visualize_comparison.py "$model"
    ((step++))
done

echo ""
echo "=========================================="
echo "All done!"
echo "=========================================="
