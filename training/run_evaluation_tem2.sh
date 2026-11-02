#!/bin/bash
# Evaluate nnUNet models on TEM2 at multiple pixel sizes.
#
# Models: witness + multires (models 1 and 2)
# Native px: 0.00493 um/px — finer resolutions require upsampling
# GPU: CUDA_VISIBLE_DEVICES=0
# Logs: ~/output_eval_tem2_{model}.log

set -e

DATA_DIR="${HOME}/resinv_exp/data/TEM2"
NNUNET_RESULTS="${HOME}/nnunet_results"
OUTPUT_DIR="${HOME}/resinv_exp/results_nnunet_tem2"
SCRIPTS_DIR="${HOME}/resinv_exp/scripts/training"

mkdir -p "${OUTPUT_DIR}"

echo "======================================================"
echo " nnUNet evaluation — TEM2"
echo " Output: ${OUTPUT_DIR}"
echo "======================================================"

echo ""
echo "=== Model 1: witness ==="
CUDA_VISIBLE_DEVICES=0 python "${SCRIPTS_DIR}/evaluate_nnunet.py" \
    --model-dir "${NNUNET_RESULTS}/Dataset001_TEM_witness/nnUNetTrainer__nnUNetPlans__2d" \
    --model-name witness \
    --data-dir "${DATA_DIR}" \
    --original-px 0.00493 \
    --subjects sub-370 sub-372 sub-373C sub-374 sub-375 \
    --gt-only \
    --output-dir "${OUTPUT_DIR}" \
    --gpu-id 0 \
    2>&1 | tee ~/output_eval_tem2_witness.log

echo ""
echo "=== Model 2: multires ==="
CUDA_VISIBLE_DEVICES=0 python "${SCRIPTS_DIR}/evaluate_nnunet.py" \
    --model-dir "${NNUNET_RESULTS}/Dataset002_TEM_multires/nnUNetTrainer__nnUNetPlans__2d" \
    --model-name multires \
    --data-dir "${DATA_DIR}" \
    --original-px 0.00493 \
    --subjects sub-370 sub-372 sub-373C sub-374 sub-375 \
    --gt-only \
    --output-dir "${OUTPUT_DIR}" \
    --gpu-id 0 \
    2>&1 | tee ~/output_eval_tem2_multires.log

echo ""
echo "=== Done ==="
echo "Retrieve with:"
echo "  rsync -avz yolaa@tassan.neuro.polymtl.ca:${OUTPUT_DIR}/ /Users/yolaatar/Developer/ADS/resinv/results_nnunet_tem2/"
