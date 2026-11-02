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

DATA_DIR="${HOME}/resinv_exp/data/TEM1"
NNUNET_RESULTS="${HOME}/nnunet_results"
OUTPUT_DIR="${HOME}/resinv_exp/results_nnunet"
SCRIPTS_DIR="${HOME}/resinv_exp/scripts/training"

mkdir -p "${OUTPUT_DIR}"

echo "======================================================"
echo " nnUNet evaluation — TEM1 test set"
echo " Output: ${OUTPUT_DIR}"
echo "======================================================"

echo ""
echo "=== Model 1: witness ==="
CUDA_VISIBLE_DEVICES=0 python "${SCRIPTS_DIR}/evaluate_nnunet.py" \
    --model-dir "${NNUNET_RESULTS}/Dataset001_TEM_witness/nnUNetTrainer__nnUNetPlans__2d" \
    --model-name witness \
    --data-dir "${DATA_DIR}" \
    --split-file "${HOME}/subject_split_tem1.json" \
    --output-dir "${OUTPUT_DIR}" \
    --gpu-id 0 \
    2>&1 | tee ~/output_eval_witness.log

echo ""
echo "=== Model 2: multires ==="
CUDA_VISIBLE_DEVICES=0 python "${SCRIPTS_DIR}/evaluate_nnunet.py" \
    --model-dir "${NNUNET_RESULTS}/Dataset002_TEM_multires/nnUNetTrainer__nnUNetPlans__2d" \
    --model-name multires \
    --data-dir "${DATA_DIR}" \
    --split-file "${HOME}/subject_split_tem1.json" \
    --output-dir "${OUTPUT_DIR}" \
    --gpu-id 0 \
    2>&1 | tee ~/output_eval_multires.log

echo ""
echo "=== Done ==="
echo "Retrieve results with:"
echo "  rsync -avz yolaa@tassan.neuro.polymtl.ca:${OUTPUT_DIR}/ /Users/yolaatar/Developer/ADS/resinv/results_nnunet/"
