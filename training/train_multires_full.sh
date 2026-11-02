#!/bin/bash
# Train full multi-resolution model: TEM1 (100%) + TEM2 (all annotated), standard nnUNet trainer.
#
# vs. train_multires.sh (model 2):
#   - TEM1: all subjects used (no 80/20 split)
#   - TEM2: all annotated subjects added
#   - Dataset ID: 5  (Dataset005_TEM12_multires)
#
# Fold: all (uses 100% of cases for gradient updates, no internal val split)
# GPU: CUDA_VISIBLE_DEVICES=1
# Log: ~/output_multires_full.log

set -e

DATASET_ID=5
TEM1_DIR="${HOME}/duke/temp/yolaatar/resinv_exp/data/TEM1"
TEM2_DIR="${HOME}/duke/temp/yolaatar/resinv_exp/data/TEM2/001350"
BASE_DIR="${HOME}/duke/temp/yolaatar/nnunet_resinv"
SCRIPTS_DIR="${HOME}/resinv_exp/scripts/training"

export nnUNet_raw="${BASE_DIR}/nnUNet_raw"
export nnUNet_preprocessed="/tmp/yolaatar/nnunet_preprocessed"
export nnUNet_results="${BASE_DIR}/nnUNet_results"

mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}"

echo "======================================================"
echo " Full multi-resolution model training"
echo " TEM1: ${TEM1_DIR} (100% subjects)"
echo " TEM2: ${TEM2_DIR} (all annotated subjects)"
echo " Extra resolutions: 0.007, 0.01, 0.016 um/px"
echo "======================================================"

# Step 1: Prepare dataset
echo ""
echo "=== Step 1: Preparing nnUNet dataset ==="
python "${SCRIPTS_DIR}/prepare_dataset_multires_full.py" \
    --tem1-dir "${TEM1_DIR}" \
    --tem2-dir "${TEM2_DIR}" \
    --nnunet-raw "${nnUNet_raw}"

# Step 2: Plan and preprocess (2D only)
echo ""
echo "=== Step 2: Planning and preprocessing ==="
nnUNetv2_plan_and_preprocess \
    -d ${DATASET_ID} \
    -c 2d \
    --verify_dataset_integrity

# Step 3: Train (fold all, 2D)
echo ""
echo "=== Step 3: Training (fold all, 2D) ==="
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train \
    ${DATASET_ID} \
    2d \
    all \
    2>&1 | tee ~/output_multires_full.log

echo ""
echo "=== Done ==="
echo "Checkpoint: ${nnUNet_results}/Dataset005_TEM12_multires/nnUNetTrainer__nnUNetPlans__2d/fold_all/checkpoint_final.pth"
