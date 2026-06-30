#!/bin/bash
#SBATCH --job-name=eval_multires_full
#SBATCH --output=%x_%j.log
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=def-jcohen   # adjust to your Alliance account

# ---------------------------------------------------------------------------
# Paths — adjust to your Vulcan setup
# ---------------------------------------------------------------------------
SCRIPTS_DIR="${HOME}/resinv_exp/scripts/training"
TEM1_DIR="${SCRATCH}/resinv_exp/data/TEM1"
TEM2_DIR="${SCRATCH}/resinv_exp/data/TEM2/001350"
MODEL_DIR="${SCRATCH}/resinv_exp/nnunet_results/Dataset005_TEM12_multires/nnUNetTrainer__nnUNetPlans__2d"
OUTPUT_DIR="${SCRATCH}/resinv_exp/results_multires_full"
# ---------------------------------------------------------------------------

mkdir -p "${OUTPUT_DIR}"

echo "======================================================"
echo " Eval: multires_full (Dataset005, fold_all)"
echo " TEM1: ${TEM1_DIR}"
echo " TEM2: ${TEM2_DIR}"
echo " Model: ${MODEL_DIR}"
echo " Output: ${OUTPUT_DIR}"
echo "======================================================"

# TEM1 — no split file (model trained on 100%, eval on all subjects)
echo ""
echo "=== TEM1 ==="
python "${SCRIPTS_DIR}/evaluate_nnunet.py" \
    --model-dir "${MODEL_DIR}" \
    --model-name multires_full \
    --data-dir "${TEM1_DIR}" \
    --original-px 0.00236 \
    --output-dir "${OUTPUT_DIR}" \
    --checkpoint checkpoint_final.pth \
    --gpu-id 0

# TEM2 — all annotated subjects (unlabeled images auto-skipped at metric stage)
echo ""
echo "=== TEM2 ==="
python "${SCRIPTS_DIR}/evaluate_nnunet.py" \
    --model-dir "${MODEL_DIR}" \
    --model-name multires_full \
    --data-dir "${TEM2_DIR}" \
    --original-px 0.00493 \
    --output-dir "${OUTPUT_DIR}" \
    --checkpoint checkpoint_final.pth \
    --gpu-id 0

echo ""
echo "=== Done ==="
echo "Retrieve results:"
echo "  rsync -avz \${USER}@vulcan.alliancecan.ca:${OUTPUT_DIR}/ ~/path/to/local/results_multires_full/"
