#!/bin/bash
# Train DA5 model: nnUNetTrainerDA5, TEM1 single resolution (model 3/4)
#
# Same dataset as witness (Dataset001_TEM_witness, already prepared).
# Only difference: DA5 trainer with more aggressive augmentation.
#
# GPU: CUDA_VISIBLE_DEVICES=0
# Log: ~/output_da5.log

set -e

DATASET_ID=1
BASE_DIR="${HOME}/duke/temp/yolaatar/nnunet_resinv"

export nnUNet_raw="${BASE_DIR}/nnUNet_raw"
export nnUNet_preprocessed="/tmp/yolaatar/nnunet_preprocessed"
export nnUNet_results="${BASE_DIR}/nnUNet_results"

mkdir -p "${nnUNet_preprocessed}"

echo "======================================================"
echo " DA5 model training (model 3)"
echo " Dataset: Dataset001_TEM_witness (already preprocessed)"
echo " Trainer: nnUNetTrainerDA5"
echo "======================================================"

# Preprocessed data may still be in /tmp from witness training.
# If not, rerun preprocessing (no dataset prep needed, same raw data).
if [ ! -f "${nnUNet_preprocessed}/Dataset001_TEM_witness/nnUNetPlans.json" ]; then
    echo "=== Preprocessing (plans not found in /tmp) ==="
    nnUNetv2_plan_and_preprocess -d ${DATASET_ID} -c 2d
fi

echo ""
echo "=== Training (fold 0, 2D, DA5 trainer) ==="
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train \
    ${DATASET_ID} \
    2d \
    0 \
    --tr nnUNetTrainerDA5 \
    2>&1 | tee ~/output_da5.log

echo ""
echo "=== Done ==="
echo "Checkpoint: ${nnUNet_results}/Dataset001_TEM_witness/nnUNetTrainerDA5__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
