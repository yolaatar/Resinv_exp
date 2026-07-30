#!/bin/bash
# One-time setup of a fresh venv for the v2 (nnUNet 2.8.1) retrains, on tassan.
#
# venv_resinv (2.2.1) is left untouched so old checkpoints can still be re-evaluated
# from their original environment.
#
# nnunetv2==2.8.1 and an unpinned torch match ADS's current pyproject.toml
# (axondeepseg/axondeepseg@357873c, "Update torch (and nnunetv2) + add Apple MPS support").
#
# Usage: bash setup_env_v2.sh

set -e

python3 -m venv ~/resinv_exp/venv_resinv_v2
source ~/resinv_exp/venv_resinv_v2/bin/activate

pip install --upgrade pip
pip install nnunetv2==2.8.1 torch
pip install pillow numpy scikit-image scikit-learn pandas matplotlib monai

echo ""
echo "=== Installed versions ==="
python -c "import torch, nnunetv2; print('torch', torch.__version__); print('nnunetv2', nnunetv2.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

echo ""
echo "=== venv_resinv_v2 ready ==="
echo "Activate with: source ~/resinv_exp/venv_resinv_v2/bin/activate"
echo ""
echo "NOTE: venv_resinv (2.2.1) needed 3 sed patches for PyTorch/nnUNet compat"
echo "(see run_commands.md). Only re-apply these to venv_resinv_v2 if training"
echo "actually hits the same errors — they may already be fixed upstream in 2.8.1:"
echo "  1. polylr verbose arg   -> nnunetv2/training/lr_scheduler/polylr.py"
echo "  2. weights_only (train) -> nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py"
echo "  3. weights_only (infer) -> nnunetv2/inference/predict_from_raw_data.py"
