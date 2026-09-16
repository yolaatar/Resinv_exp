#!/usr/bin/env python3
"""Concatenate per-image axon_counts.csv files (written by axondeepseg_count,
one per {image}/predictions/ folder, one row per pixel size) into a single
CSV for a TEM3 model results directory.

Usage:
    python aggregate_axon_counts_tem3.py --results-dir ~/resinv_exp/results_tem3/armand_uaxon \
        --out ~/resinv_exp/results_tem3/armand_uaxon/axon_counts_all.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True,
                     help="e.g. ~/resinv_exp/results_tem3/armand_uaxon")
    ap.add_argument("--out", type=Path, default=None,
                     help="Default: {results-dir}/axon_counts_all.csv")
    args = ap.parse_args()

    out_path = args.out or (args.results_dir / "axon_counts_all.csv")

    files = sorted(args.results_dir.glob("*/predictions/axon_counts.csv"))
    if not files:
        print(f"No axon_counts.csv files found under {args.results_dir}")
        return

    dfs = [pd.read_csv(f) for f in files]
    summary = pd.concat(dfs, ignore_index=True)
    summary.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  ({len(summary)} rows from {len(files)} images)")


if __name__ == "__main__":
    main()
