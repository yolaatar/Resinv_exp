#!/bin/bash
# Evaluate nnUNet v2 (nnUNet 2.8.1) models on TEM2 at multiple pixel sizes.
#
# Models: multires_v2 (Dataset003, fold_0) + multires_full_v2 (Dataset006, fold_all)
# Native px: 0.00493 um/px — finer resolutions require upsampling
# GPU: CUDA_VISIBLE_DEVICES=1
# Logs: ~/output_eval_tem2_{model}.log

set -e

DATA_DIR="${HOME}/resinv_exp/data/TEM2"
NNUNET_RESULTS="${HOME}/resinv_exp/nnunet_resinv_v2/nnUNet_results"
OUTPUT_DIR="${HOME}/resinv_exp/results_nnunet_tem2"
SCRIPTS_DIR="${HOME}/resinv_exp/scripts/training"

mkdir -p "${OUTPUT_DIR}"

echo "======================================================"
echo " nnUNet v2 evaluation — TEM2"
echo " Output: ${OUTPUT_DIR}"
echo "======================================================"

echo ""
echo "=== Model 1: multires_v2 (fold_0) ==="
CUDA_VISIBLE_DEVICES=1 python "${SCRIPTS_DIR}/evaluate_nnunet.py" \
    --model-dir "${NNUNET_RESULTS}/Dataset003_TEM_multires_v2/nnUNetTrainer__nnUNetPlans__2d" \
    --model-name multires_v2 \
    --data-dir "${DATA_DIR}" \
    --original-px 0.00493 \
    --subjects sub-370 sub-372 sub-373C sub-374 sub-375 \
    --gt-only \
    --fold 0 \
    --checkpoint checkpoint_best.pth \
    --output-dir "${OUTPUT_DIR}" \
    --gpu-id 0 \
    2>&1 | tee ~/output_eval_tem2_multires_v2.log

echo ""
echo "=== Model 2: multires_full_v2 (fold_all) ==="
CUDA_VISIBLE_DEVICES=1 python "${SCRIPTS_DIR}/evaluate_nnunet.py" \
    --model-dir "${NNUNET_RESULTS}/Dataset006_TEM12_multires_v2/nnUNetTrainer__nnUNetPlans__2d" \
    --model-name multires_full_v2 \
    --data-dir "${DATA_DIR}" \
    --original-px 0.00493 \
    --subjects sub-370 sub-372 sub-373C sub-374 sub-375 \
    --gt-only \
    --fold all \
    --checkpoint checkpoint_final.pth \
    --output-dir "${OUTPUT_DIR}" \
    --gpu-id 0 \
    2>&1 | tee ~/output_eval_tem2_multires_full_v2.log

echo ""
echo "=== Done ==="
echo "Retrieve with:"
echo "  rsync -avz yolaa@tassan.neuro.polymtl.ca:${OUTPUT_DIR}/ /Users/yolaatar/Developer/ADS/resinv/results_nnunet_tem2/"
