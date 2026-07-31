#!/bin/bash
# Launch the held-out TEM2 test set eval across both GPUs in parallel, via tmux.
#
#   tmux session eval_testset_tem2_gpu0 -> witness, multires, da5, da5_multires   (GPU 0)
#   tmux session eval_testset_tem2_gpu1 -> multires_full, multires_v2, multires_full_v2 (GPU 1)
#
# Check on them with:
#   tmux attach -t eval_testset_tem2_gpu0
#   tmux attach -t eval_testset_tem2_gpu1
#
# Usage: bash run_evaluation_testset_tem2_parallel.sh

set -e

SCRIPTS_DIR="${HOME}/resinv_exp/scripts/training"

tmux new -d -s eval_testset_tem2_gpu0 \
    "bash ${SCRIPTS_DIR}/run_evaluation_testset_tem2.sh 0 witness multires da5 da5_multires"

tmux new -d -s eval_testset_tem2_gpu1 \
    "bash ${SCRIPTS_DIR}/run_evaluation_testset_tem2.sh 1 multires_full multires_v2 multires_full_v2"

echo "Launched:"
tmux ls
