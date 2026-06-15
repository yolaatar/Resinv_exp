#!/usr/bin/env python3
"""
Recompute Dice metrics from existing prediction PNGs using MONAI.

Run this locally after rsync-ing results from the cluster — no GPU needed.

For axon/myelin labels: Dice is computed against manual GT masks from
  {data_dir}/derivatives/labels/{subject}/micr/{img_name}_seg-{label}-manual.png

For uaxon (no GT available): Dice is computed against the model prediction
  at the closest pixel size to training resolution (0.00493 μm/px).

Usage:
    python recompute_metrics.py --results-dir ./results --data-dir ./data/TEM1
    python recompute_metrics.py --results-dir ./results   # uaxon-only mode
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.metrics import compute_dice
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

TRAINING_PX = 0.00493
ORIGINAL_PX = 0.0018625
# Labels with GT available — compare against manual masks
GT_LABELS = {"axon", "myelin"}
# Labels to ignore entirely
SKIP_LABELS = {"nuclei", "process", "axonmyelin"}


def load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


def resample_mask(img: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    from skimage.transform import resize
    return resize(img, target_shape, order=0, preserve_range=True,
                  anti_aliasing=False).astype(img.dtype)


def dice_monai(a: np.ndarray, b: np.ndarray) -> float:
    a_t = torch.from_numpy((a > 0).astype(np.uint8)).unsqueeze(0).unsqueeze(0)
    b_t = torch.from_numpy((b > 0).astype(np.uint8)).unsqueeze(0).unsqueeze(0)
    score = compute_dice(a_t, b_t, include_background=True, ignore_empty=True)
    val = float(score[0, 0])
    return 1.0 if np.isnan(val) else val


def parse_px(filename: str) -> float | None:
    m = re.search(r"_px([\d.]+)um_", filename)
    return float(m.group(1)) if m else None


def find_gt_mask(img_name: str, label: str, data_dir: Path) -> Path | None:
    """Look up a manual GT mask in the BIDS derivatives tree."""
    # Subject is the first component of the BIDS image name (e.g. sub-nyuMouse30)
    subject = img_name.split("_")[0]
    gt_path = data_dir / "derivatives" / "labels" / subject / "micr" / f"{img_name}_seg-{label}-manual.png"
    return gt_path if gt_path.exists() else None


def detect_active_labels(pred_dir: Path, ref_px: float) -> list[str]:
    """
    Return labels with non-trivial predictions at the reference resolution.
    Skips labels in SKIP_LABELS and those covering >50% of the image.
    """
    all_labels = sorted({
        re.search(r"_seg-(.+)\.png", p.name).group(1)
        for p in pred_dir.glob("*_seg-*.png")
        if re.search(r"_seg-(.+)\.png", p.name)
    })
    ref_tag = f"_px{ref_px:.7g}um_"
    active = []
    for label in all_labels:
        if label in SKIP_LABELS:
            continue
        candidates = list(pred_dir.glob(f"*{ref_tag}seg-{label}.png"))
        if not candidates:
            candidates = [p for p in pred_dir.glob(f"*_seg-{label}.png")
                          if parse_px(p.name) is not None and abs(parse_px(p.name) - ref_px) < 1e-7]
        if not candidates:
            continue
        arr = load_gray(candidates[0])
        coverage = (arr > 0).mean()
        if 0 < coverage < 0.5:
            active.append(label)
    return active


def recompute_image(img_dir: Path, model_name: str, data_dir: Path | None) -> pd.DataFrame:
    pred_dir = img_dir / "predictions"
    img_name = img_dir.name
    if not pred_dir.exists():
        return pd.DataFrame()

    all_pred_files = list(pred_dir.glob("*_seg-*.png"))
    px_set = sorted({parse_px(p.name) for p in all_pred_files if parse_px(p.name)})
    if not px_set:
        return pd.DataFrame()

    ref_px = min(px_set, key=lambda x: abs(x - TRAINING_PX))
    active_labels = detect_active_labels(pred_dir, ref_px)

    if not active_labels:
        print(f"  No active labels found, skipping.")
        return pd.DataFrame()

    print(f"  Labels: {active_labels}")

    # Build reference mask per label: GT if available, else prediction at ref_px
    ref_masks = {}
    ref_sources = {}  # track where the reference came from for logging
    for label in active_labels:
        gt_path = find_gt_mask(img_name, label, data_dir) if (data_dir and label in GT_LABELS) else None
        if gt_path is not None:
            ref_masks[label] = load_gray(gt_path)
            ref_sources[label] = "gt"
            print(f"  {label}: using GT from {gt_path.name}")
        else:
            candidates = [p for p in pred_dir.glob(f"*_seg-{label}.png")
                          if parse_px(p.name) is not None and abs(parse_px(p.name) - ref_px) < 1e-7]
            if candidates:
                ref_masks[label] = load_gray(candidates[0])
                ref_sources[label] = "pred"
                if label in GT_LABELS and data_dir:
                    print(f"  {label}: GT not found, falling back to prediction at {ref_px} μm/px")
                else:
                    print(f"  {label}: using prediction at {ref_px} μm/px as reference")

    if not ref_masks:
        print(f"  No reference masks available, skipping.")
        return pd.DataFrame()

    # Each label may have a different reference shape (GT vs pred dimensions differ)
    ref_shapes = {label: ref_masks[label].shape for label in ref_masks}

    rows = []
    for px in px_set:
        sample = next((p for p in pred_dir.glob("*_seg-*.png")
                       if parse_px(p.name) is not None and abs(parse_px(p.name) - px) < 1e-7), None)
        if sample is None:
            continue
        pred_h, pred_w = load_gray(sample).shape

        row = {
            "image": img_name,
            "model": model_name,
            "pixel_size_um": px,
            "scale_factor": round(ORIGINAL_PX / px, 6),
            "image_width": pred_w,
            "image_height": pred_h,
            "reference_px": ref_px,
        }

        for label in active_labels:
            if label not in ref_masks:
                continue
            ref_mask = ref_masks[label]
            ref_h, ref_w = ref_shapes[label]
            pred_path = next((p for p in pred_dir.glob(f"*_seg-{label}.png")
                              if parse_px(p.name) is not None and abs(parse_px(p.name) - px) < 1e-7), None)
            if pred_path is None:
                continue
            pred = load_gray(pred_path)

            row[f"dice_{label}_native"] = dice_monai(resample_mask(pred, (ref_h, ref_w)), ref_mask)
            row[f"dice_{label}_resized"] = dice_monai(pred, resample_mask(ref_mask, (pred_h, pred_w)))
            row[f"dice_{label}_interp_baseline"] = dice_monai(
                resample_mask(resample_mask(ref_mask, (pred_h, pred_w)), (ref_h, ref_w)), ref_mask
            )

        rows.append(row)
        label_summary = ", ".join(
            f"{l}({'GT' if ref_sources.get(l) == 'gt' else 'pred'}) native: {row.get(f'dice_{l}_native', float('nan')):.3f}"
            for l in active_labels
        )
        print(f"  {px} μm/px — {label_summary}")

    df = pd.DataFrame(rows)
    df.to_csv(img_dir / "metrics.csv", index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("./results"))
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Dataset root (BIDS). Needed to load GT masks for axon/myelin. "
                             "GT looked up at derivatives/labels/{subject}/micr/{img}_seg-{label}-manual.png")
    args = parser.parse_args()

    for model_dir in sorted(args.results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        print(f"\n{'='*60}\nModel: {model_name}")

        all_dfs = []
        for img_dir in sorted(model_dir.iterdir()):
            if not img_dir.is_dir():
                continue
            print(f"\nImage: {img_dir.name}")
            df = recompute_image(img_dir, model_name, args.data_dir)
            if not df.empty:
                all_dfs.append(df)

        if all_dfs:
            summary = pd.concat(all_dfs, ignore_index=True)
            summary.to_csv(model_dir / "results.csv", index=False)
            print(f"\nSaved: {model_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
