#!/bin/bash
# Install the ResInv custom trainers into the nnUNet 2.8.1 venv.
#
# nnUNetv2_train only finds trainer classes that live inside the installed nnunetv2 package
# (it searches nnunetv2/training/nnUNetTrainer recursively), so the file has to be copied in.
# Re-run after editing nnUNetTrainerResInv.py or after reinstalling nnunetv2.
#
# Usage: bash install_custom_trainers.sh [venv_dir]   (default ~/resinv_exp/venv_resinv_v2)

set -e

VENV="${1:-${HOME}/resinv_exp/venv_resinv_v2}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${VENV}/bin/python"

[ -x "${PY}" ] || { echo "No python at ${PY}"; exit 1; }

TRAINER_ROOT="$("${PY}" -c 'import nnunetv2, os; print(os.path.join(os.path.dirname(nnunetv2.__file__), "training", "nnUNetTrainer"))')"
DEST="${TRAINER_ROOT}/variants/resinv"
mkdir -p "${DEST}"
touch "${DEST}/__init__.py"
cp "${HERE}/nnUNetTrainerResInv.py" "${DEST}/nnUNetTrainerResInv.py"
echo "Copied trainers to ${DEST}"

# Check nnUNet can resolve each class by name, the same way nnUNetv2_train does.
"${PY}" - <<'EOF'
import os
import nnunetv2
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
root = os.path.join(os.path.dirname(nnunetv2.__file__), "training", "nnUNetTrainer")
for name in ["nnUNetTrainerScaleAug2p5", "nnUNetTrainerScaleAug2p5_p05", "nnUNetTrainerDA5"]:
    cls = recursive_find_python_class(root, name, "nnunetv2.training.nnUNetTrainer")
    assert cls is not None, f"nnUNet cannot find {name}"
    print(f"  OK {name} -> {cls.__module__}")
EOF
