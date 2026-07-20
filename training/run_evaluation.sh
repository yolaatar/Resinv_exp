#!/bin/bash
# Evaluate nnUNet models on TEM1 test set at multiple pixel sizes.
#
# Models evaluated:
#   1. witness  — Dataset001, nnUNetTrainer
#   2. multires — Dataset002, nnUNetTrainer
#
# GPU: CUDA_VISIBLE_DEVICES=0 (shows as cuda:0 inside scripts)
# Logs: ~/output_eval_{model}.log

set -e

DATA_DIR="${HOME}/duke/temp/yolaatar/resinv_exp/data/TEM1"
BASE_DIR="${HOME}/duke/temp/yolaatar/nnunet_resinv"
OUTPUT_DIR="${HOME}/duke/temp/yolaatar/resinv_exp/results_nnunet"
SCRIPTS_DIR="${HOME}/resinv_exp/scripts/training"

export nnUNet_raw="${BASE_DIR}/nnUNet_raw"
export nnUNet_results="${BASE_DIR}/nnUNet_results"

mkdir -p "${OUTPUT_DIR}"

echo "======================================================"
echo " nnUNet evaluation — TEM1 test set"
echo " Output: ${OUTPUT_DIR}"
echo "======================================================"

echo ""
echo "=== Model 1: witness ==="
CUDA_VISIBLE_DEVICES=0 python "${SCRIPTS_DIR}/evaluate_nnunet.py" \
    --model-dir "${nnUNet_results}/Dataset001_TEM_witness/nnUNetTrainer__nnUNetPlans__2d" \
    --model-name witness \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-images 50 \
    --gpu-id 0 \
    2>&1 | tee ~/output_eval_witness.log

echo ""
echo "=== Model 2: multires ==="
CUDA_VISIBLE_DEVICES=0 python "${SCRIPTS_DIR}/evaluate_nnunet.py" \
    --model-dir "${nnUNet_results}/Dataset002_TEM_multires/nnUNetTrainer__nnUNetPlans__2d" \
    --model-name multires \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-images 50 \
    --gpu-id 0 \
    2>&1 | tee ~/output_eval_multires.log

echo ""
echo "=== Done ==="
echo "Retrieve results with:"
echo "  rsync -avz yolaa@tassan.neuro.polymtl.ca:${OUTPUT_DIR}/ /Users/yolaatar/Developer/ADS/resinv/results_nnunet/"
