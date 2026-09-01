#!/usr/bin/env python3
"""Extract manual axon/uaxon counts from TEM3's keypoints JSON sidecars.

Each derivatives/keypoints/{subject}/micr/{subject}_{sample}_keypoints.json
holds {"Myelinated": [[x,y], ...], "Unmyelinated": [[x,y], ...]}, one point
per manually-identified instance. The manual count is just the length of
each list -- this is the ground truth to compare predicted counts against
for TEM3 (no segmentation GT / Dice for this dataset).

Usage:
    python extract_manual_counts_tem3.py \
        --data-dir ~/resinv_exp/data/data_axondeepseg_stanford_app \
        --out manual_counts_tem3.csv
"""

import argparse
import csv
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True,
                     help="TEM3 dataset root (BIDS structure)")
    ap.add_argument("--out", type=Path, default=Path("manual_counts_tem3.csv"))
    args = ap.parse_args()

    kp_root = args.data_dir / "derivatives" / "keypoints"
    rows = []
    for kp_file in sorted(kp_root.glob("*/micr/*_keypoints.json")):
        data = json.loads(kp_file.read_text())
        image = kp_file.stem.replace("_keypoints", "") + "_TEM"
        rows.append({
            "image": image,
            "manual_axon_count": len(data.get("Myelinated", [])),
            "manual_uaxon_count": len(data.get("Unmyelinated", [])),
        })

    if not rows:
        print(f"No keypoints files found under {kp_root}")
        return

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "manual_axon_count", "manual_uaxon_count"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
