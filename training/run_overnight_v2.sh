#!/bin/bash
# Launch both v2 (nnUNet 2.8.1) retrains overnight, in parallel, on tassan.
#
#   tmux session multires_v2       -> train_multires_v2.sh       (TEM1 only,      GPU 0, fold_0)
#   tmux session multires_full_v2  -> train_multires_full_v2.sh  (TEM1 + TEM2,    GPU 1, fold_all)
#
# Both sessions survive SSH disconnects. Check on them with:
#   tmux attach -t multires_v2
#   tmux attach -t multires_full_v2
#   tail -f ~/output_multires_v2.log
#   tail -f ~/output_multires_full_v2.log
#
# Usage: bash run_overnight_v2.sh

set -e

SCRIPTS_DIR="${HOME}/resinv_exp/scripts/training"
VENV="${HOME}/resinv_exp/venv_resinv_v2/bin/activate"

tmux new -d -s multires_v2 \
    "source ${VENV} && bash ${SCRIPTS_DIR}/train_multires_v2.sh"

tmux new -d -s multires_full_v2 \
    "source ${VENV} && bash ${SCRIPTS_DIR}/train_multires_full_v2.sh"

echo "Launched:"
tmux ls
