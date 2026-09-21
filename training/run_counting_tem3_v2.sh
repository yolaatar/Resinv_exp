#!/usr/bin/env bash
# Same as run_counting_tem3.sh but with a custom filter config and a distinct
# output filename, so it doesn't collide with the default-config run's
# resume-skip logic (which would otherwise skip every folder since
# axon_counts.csv already exists from the first pass).
#
# axondeepseg_filter (no -o) always regenerates *_filtered.xlsx from the raw
# *_morphometrics.xlsx with the new config -- overwrites the old (no-op)
# _filtered.xlsx from the default-config run, raw files untouched.
#
# Usage:
#   bash run_counting_tem3_v2.sh CONFIG_PATH [RESULTS_ROOT]
# Default RESULTS_ROOT: ~/resinv_exp/results_tem3

set -euo pipefail

CONFIG_PATH="${1:?Usage: run_counting_tem3_v2.sh CONFIG_PATH [RESULTS_ROOT]}"
RESULTS_ROOT="${2:-$HOME/resinv_exp/results_tem3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for MODEL in armand_uaxon armand_control; do
    RESULTS_DIR="${RESULTS_ROOT}/${MODEL}"
    if [ ! -d "$RESULTS_DIR" ]; then
        echo "SKIP: ${RESULTS_DIR} does not exist"
        continue
    fi

    echo "======================================================"
    echo " ${MODEL}: filtering (custom config) + counting (v2)"
    echo "======================================================"

    for d in "${RESULTS_DIR}"/*/predictions; do
        [ -d "$d" ] || continue
        out="${d}/axon_counts_v2.csv"
        if [ -f "$out" ]; then
            echo "skip $(dirname "$d") (already done)"
            continue
        fi
        echo "=== $(basename "$(dirname "$d")") ==="
        axondeepseg_filter -i "$d" -c "$CONFIG_PATH"
        axondeepseg_count -i "$d" -o "$out"
    done

    echo "Aggregating ${MODEL} (v2)..."
    python "${SCRIPT_DIR}/aggregate_axon_counts_tem3.py" \
        --results-dir "${RESULTS_DIR}" \
        --pattern "axon_counts_v2.csv" \
        --out "${RESULTS_DIR}/axon_counts_all_v2.csv"
done

echo "Done."
