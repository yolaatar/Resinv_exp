#!/usr/bin/env python3
"""
Recompute Dice metrics from existing prediction PNGs using MONAI.

Run this locally after rsync-ing results from the cluster — no GPU needed.

Usage:
    python recompute_metrics.py --results-dir ./results
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
# Labels to ignore: nuclei tends to cover 80%+ of the image (catch-all class),
# process is essentially empty in corpus callosum data.
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


def detect_active_labels(pred_dir: Path, ref_px: float) -> list[str]:
    """
    Return labels that have non-trivial predictions at the reference resolution.
    Skips labels in SKIP_LABELS and those covering >50% of the image (catch-all classes).
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
            # Try approximate match
            candidates = [p for p in pred_dir.glob(f"*_seg-{label}.png")
                          if abs(parse_px(p.name) - ref_px) < 1e-7]
        if not candidates:
            continue
        arr = load_gray(candidates[0])
        coverage = (arr > 0).mean()
        if coverage > 0 and coverage < 0.5:
            active.append(label)
    return active


def compute_label_metrics(pred_dir: Path, label: str, ref_mask: np.ndarray,
                           px_files: dict[float, Path], ref_px: float) -> dict:
    """Compute native, resized, and interp Dice for one label across all pixel sizes."""
    ref_h, ref_w = ref_mask.shape
    results = {}
    for px, primary_path in sorted(px_files.items()):
        pred_path = pred_dir / primary_path.name.replace(
            f"_seg-{list(px_files.values())[0].name.split('_seg-')[1]}",
            f"_seg-{label}.png"
        )
        # Rebuild path from scratch to avoid string replace issues
        stem = re.sub(r"_seg-.+\.png$", f"_seg-{label}.png", primary_path.name)
        pred_path = pred_dir / stem
        if not pred_path.exists():
            continue

        pred = load_gray(pred_path)
        pred_h, pred_w = pred.shape

        native = dice_monai(resample_mask(pred, (ref_h, ref_w)), ref_mask)
        resized = dice_monai(pred, resample_mask(ref_mask, (pred_h, pred_w)))
        interp = dice_monai(
            resample_mask(resample_mask(ref_mask, (pred_h, pred_w)), (ref_h, ref_w)),
            ref_mask
        )
        results[px] = (native, resized, interp)
    return results


def recompute_image(img_dir: Path, model_name: str) -> pd.DataFrame:
    pred_dir = img_dir / "predictions"
    img_name = img_dir.name
    if not pred_dir.exists():
        return pd.DataFrame()

    # Find all pixel sizes using any label as anchor
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

    # Load reference masks for each label
    ref_masks = {}
    for label in active_labels:
        candidates = [p for p in pred_dir.glob(f"*_seg-{label}.png")
                      if abs(parse_px(p.name) - ref_px) < 1e-7]
        if candidates:
            ref_masks[label] = load_gray(candidates[0])

    ref_h, ref_w = next(iter(ref_masks.values())).shape

    rows = []
    for px in px_set:
        # Get image dimensions from any existing prediction
        sample = next((p for p in pred_dir.glob(f"*_seg-*.png")
                       if abs(parse_px(p.name) - px) < 1e-7), None)
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
            pred_path = next((p for p in pred_dir.glob(f"*_seg-{label}.png")
                              if abs(parse_px(p.name) - px) < 1e-7), None)
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
            f"{l} native: {row.get(f'dice_{l}_native', float('nan')):.3f}"
            for l in active_labels
        )
        print(f"  {px} μm/px — {label_summary}")

    df = pd.DataFrame(rows)
    df.to_csv(img_dir / "metrics.csv", index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("./results"))
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
            df = recompute_image(img_dir, model_name)
            if not df.empty:
                all_dfs.append(df)

        if all_dfs:
            summary = pd.concat(all_dfs, ignore_index=True)
            summary.to_csv(model_dir / "results.csv", index=False)
            print(f"\nSaved: {model_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
