#!/usr/bin/env python3
"""No-GPU sanity check for TEM3: verify every image resolves a native pixel
size via evaluate_nnunet.load_original_px (sidecar JSON), and report the
distribution of values found + any images that silently fell back to the
--original-px default (that would mean a naming-convention mismatch).

Usage:
    python check_tem3_pixel_sizes.py --data-dir ~/resinv_exp/data/data_axondeepseg_stanford_app
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_nnunet import find_images, load_original_px  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--fallback", type=float, default=0.00493,
                     help="Same default evaluate_nnunet.py would use, to detect silent fallbacks")
    args = ap.parse_args()

    images = find_images(args.data_dir, args.data_dir / "subject_split.json")
    print(f"Found {len(images)} raw images\n")

    px_counts = Counter()
    fell_back = []
    for img in images:
        px = load_original_px(img, args.fallback)
        px_counts[px] += 1
        # heuristic: flag as suspicious if it matches the fallback AND no sidecar exists
        subject = img.stem.split("_")[0]
        suffix = img.stem.split("_")[-1]
        has_sidecar = img.with_suffix(".json").exists() or (img.parent / f"{subject}_{suffix}.json").exists()
        if not has_sidecar:
            fell_back.append(img.name)

    print("Pixel size distribution (um/px -> image count):")
    for px, n in sorted(px_counts.items()):
        print(f"  {px}: {n}")

    if fell_back:
        print(f"\nWARNING: {len(fell_back)} images had NO sidecar JSON found, used fallback {args.fallback}:")
        for name in fell_back:
            print(f"  {name}")
    else:
        print("\nAll images resolved a pixel size from a sidecar JSON (no silent fallbacks).")


if __name__ == "__main__":
    main()
