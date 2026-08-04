#!/usr/bin/env python3
"""Concatenate per-image count_axons.py outputs into one CSV, tagging pixel size.

Usage:
    python aggregate_axon_counts.py --counts-dir axon_counts --out axon_counts_all.csv
"""

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_px(image_name: str) -> float | None:
    m = re.search(r"_px([\d.]+)um", image_name)
    return float(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts-dir", type=Path, default=Path("axon_counts"))
    parser.add_argument("--out", type=Path, default=Path("axon_counts_all.csv"))
    args = parser.parse_args()

    dfs = []
    for f in sorted(args.counts_dir.glob("*_counts.csv")):
        df = pd.read_csv(f)
        df["pixel_size_um"] = df["image"].apply(parse_px)
        dfs.append(df)

    if not dfs:
        print(f"No count files found in {args.counts_dir}")
        return

    summary = pd.concat(dfs, ignore_index=True)
    summary.to_csv(args.out, index=False)
    print(f"Saved: {args.out}  ({len(summary)} rows)")


if __name__ == "__main__":
    main()
