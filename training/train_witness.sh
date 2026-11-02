#!/bin/bash
# Train witness model: standard nnUNet, TEM1, single resolution (model 1/3)
#
# Dependencies: nnunetv2==2.2.1, Pillow, numpy
# GPU: single GPU, CUDA_VISIBLE_DEVICES=0
#
# Runtime estimate: ~8-12h on a single modern GPU (1000 epochs, 2D, ~126 images)
#
# Usage: bash train_witness.sh
# Log:   ~/output_witness.log

set -e

DATASET_ID=1
DATA_DIR="${HOME}/duke/temp/yolaatar/resinv_exp/data/TEM1"
BASE_DIR="${HOME}/duke/temp/yolaatar/nnunet_resinv"
SCRIPTS_DIR="${HOME}/resinv_exp/scripts/training"

# Raw data stays on duke (source only, read once during preprocessing)
# Preprocessed and results go to local /tmp to avoid network I/O bottleneck during training
export nnUNet_raw="${BASE_DIR}/nnUNet_raw"
export nnUNet_preprocessed="/tmp/yolaatar/nnunet_preprocessed"
export nnUNet_results="${BASE_DIR}/nnUNet_results"  # duke: checkpoints written infrequently, network latency ok

mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}"

echo "======================================================"
echo " Witness model training"
echo " Dataset: ${DATA_DIR}"
echo " nnUNet_raw: ${nnUNet_raw}"
echo " nnUNet_results: ${nnUNet_results}"
echo "======================================================"

# Step 1: Prepare dataset
echo ""
echo "=== Step 1: Preparing nnUNet dataset ==="
python "${SCRIPTS_DIR}/prepare_dataset_witness.py" \
    --data-dir "${DATA_DIR}" \
    --nnunet-raw "${nnUNet_raw}"

# Step 2: Plan and preprocess (2D only)
echo ""
echo "=== Step 2: Planning and preprocessing ==="
nnUNetv2_plan_and_preprocess \
    -d ${DATASET_ID} \
    -c 2d \
    --verify_dataset_integrity

# Step 3: Train (fold 0, 2D)
echo ""
echo "=== Step 3: Training (fold 0, 2D) ==="
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train \
    ${DATASET_ID} \
    2d \
    0 \
    2>&1 | tee ~/output_witness.log

echo ""
echo "=== Done ==="
echo "Model checkpoint: ${nnUNet_results}/Dataset001_TEM_witness/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
