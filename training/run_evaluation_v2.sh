#!/bin/bash
# Evaluate nnUNet v2 (nnUNet 2.8.1) models on TEM1 test set at multiple pixel sizes.
#
# Models evaluated:
#   1. multires_v2      — Dataset003, fold_0,   checkpoint_best.pth
#   2. multires_full_v2 — Dataset006, fold_all, checkpoint_final.pth (no val split, no "best")
#
# GPU: CUDA_VISIBLE_DEVICES=0 (shows as cuda:0 inside scripts)
# Logs: ~/output_eval_{model}.log

set -e

DATA_DIR="${HOME}/resinv_exp/data/TEM1"
NNUNET_RESULTS="${HOME}/resinv_exp/nnunet_resinv_v2/nnUNet_results"
OUTPUT_DIR="${HOME}/resinv_exp/results_nnunet"
SCRIPTS_DIR="${HOME}/resinv_exp/scripts/training"

mkdir -p "${OUTPUT_DIR}"

echo "======================================================"
echo " nnUNet v2 evaluation — TEM1 test set"
echo " Output: ${OUTPUT_DIR}"
echo "======================================================"

echo ""
echo "=== Model 1: multires_v2 (fold_0) ==="
CUDA_VISIBLE_DEVICES=0 python "${SCRIPTS_DIR}/evaluate_nnunet.py" \
    --model-dir "${NNUNET_RESULTS}/Dataset003_TEM_multires_v2/nnUNetTrainer__nnUNetPlans__2d" \
    --model-name multires_v2 \
    --data-dir "${DATA_DIR}" \
    --split-file "${HOME}/subject_split_tem1.json" \
    --fold 0 \
    --checkpoint checkpoint_best.pth \
    --output-dir "${OUTPUT_DIR}" \
    --gpu-id 0 \
    2>&1 | tee ~/output_eval_multires_v2.log

echo ""
echo "=== Model 2: multires_full_v2 (fold_all) ==="
CUDA_VISIBLE_DEVICES=0 python "${SCRIPTS_DIR}/evaluate_nnunet.py" \
    --model-dir "${NNUNET_RESULTS}/Dataset006_TEM12_multires_v2/nnUNetTrainer__nnUNetPlans__2d" \
    --model-name multires_full_v2 \
    --data-dir "${DATA_DIR}" \
    --split-file "${HOME}/subject_split_tem1.json" \
    --fold all \
    --checkpoint checkpoint_final.pth \
    --output-dir "${OUTPUT_DIR}" \
    --gpu-id 0 \
    2>&1 | tee ~/output_eval_multires_full_v2.log

echo ""
echo "=== Done ==="
echo "Retrieve results with:"
echo "  rsync -avz yolaa@tassan.neuro.polymtl.ca:${OUTPUT_DIR}/ /Users/yolaatar/Developer/ADS/resinv/results_nnunet/"
