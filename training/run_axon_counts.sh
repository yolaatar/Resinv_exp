#!/usr/bin/env bash
# Run Armand's count_axons.py over every image's predictions/ folder, then aggregate.
#
# Usage (on tassan, venv_resinv activated):
#   bash ~/resinv_exp/scripts/training/run_axon_counts.sh

set -euo pipefail

RESULTS_DIR="${1:-$HOME/resinv_exp/results_armand_uaxon/armand_uaxon}"
SCRIPTS_DIR="${2:-$HOME/resinv_exp/scripts}"

cd "$RESULTS_DIR"
mkdir -p axon_counts

for d in */predictions; do
    img=$(dirname "$d")
    out="axon_counts/${img}_counts.csv"
    if [ -f "$out" ]; then
        echo "skip $img (already done)"
        continue
    fi
    echo "=== $img ==="
    python "$SCRIPTS_DIR/count_axons.py" "$d" "$out"
done

python "$SCRIPTS_DIR/aggregate_axon_counts.py" --counts-dir axon_counts --out axon_counts_all.csv
echo "Done."
