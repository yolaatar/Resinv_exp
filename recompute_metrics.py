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


def detect_labels(pred_dir: Path) -> tuple[str, str | None]:
    """Infer primary/secondary labels from existing prediction filenames."""
    labels = sorted({
        re.search(r"_seg-(.+)\.png", p.name).group(1)
        for p in pred_dir.glob("*_seg-*.png")
        if re.search(r"_seg-(.+)\.png", p.name)
    })
    # Prefer uaxon > axon as primary (unmyelinated model), myelin as secondary
    primary_order = ["uaxon", "axon"] + [l for l in labels if l not in ("uaxon", "axon", "myelin")]
    primary = next((l for l in primary_order if l in labels), labels[0] if labels else "axon")
    secondary = "myelin" if "myelin" in labels else None
    return primary, secondary


def recompute_image(img_dir: Path, model_name: str) -> pd.DataFrame:
    pred_dir = img_dir / "predictions"
    img_name = img_dir.name
    if not pred_dir.exists():
        return pd.DataFrame()

    primary, secondary = detect_labels(pred_dir)

    # Collect all pixel sizes present
    px_files: dict[float, Path] = {}
    for p in pred_dir.glob(f"*_seg-{primary}.png"):
        px = parse_px(p.name)
        if px is not None:
            px_files[px] = p

    if not px_files:
        return pd.DataFrame()

    # Reference = closest to training resolution
    ref_px = min(px_files, key=lambda x: abs(x - TRAINING_PX))
    ref_primary = load_gray(px_files[ref_px])
    ref_h, ref_w = ref_primary.shape

    ref_secondary = None
    if secondary:
        ref_sec_path = pred_dir / px_files[ref_px].name.replace(f"_seg-{primary}.png", f"_seg-{secondary}.png")
        if ref_sec_path.exists():
            ref_secondary = load_gray(ref_sec_path)

    rows = []
    for px in sorted(px_files):
        pred_primary = load_gray(px_files[px])
        pred_h, pred_w = pred_primary.shape

        pred_sec_path = pred_dir / px_files[px].name.replace(f"_seg-{primary}.png", f"_seg-{secondary}.png")
        pred_secondary = load_gray(pred_sec_path) if secondary and pred_sec_path.exists() else None

        # Native space
        pred_primary_native = resample_mask(pred_primary, (ref_h, ref_w))
        dice_primary_native = dice_monai(pred_primary_native, ref_primary)

        dice_secondary_native = None
        if pred_secondary is not None and ref_secondary is not None:
            pred_sec_native = resample_mask(pred_secondary, (ref_h, ref_w))
            dice_secondary_native = dice_monai(pred_sec_native, ref_secondary)

        # Resized space
        ref_primary_ds = resample_mask(ref_primary, (pred_h, pred_w))
        dice_primary_resized = dice_monai(pred_primary, ref_primary_ds)

        dice_secondary_resized = None
        if pred_secondary is not None and ref_secondary is not None:
            ref_sec_ds = resample_mask(ref_secondary, (pred_h, pred_w))
            dice_secondary_resized = dice_monai(pred_secondary, ref_sec_ds)

        # Interpolation baseline
        ref_primary_roundtrip = resample_mask(
            resample_mask(ref_primary, (pred_h, pred_w)), (ref_h, ref_w)
        )
        dice_primary_interp = dice_monai(ref_primary_roundtrip, ref_primary)

        dice_secondary_interp = None
        if ref_secondary is not None:
            ref_sec_roundtrip = resample_mask(
                resample_mask(ref_secondary, (pred_h, pred_w)), (ref_h, ref_w)
            )
            dice_secondary_interp = dice_monai(ref_sec_roundtrip, ref_secondary)

        rows.append({
            "image": img_name,
            "model": model_name,
            "pixel_size_um": px,
            "scale_factor": round(0.0018625 / px, 6),
            "image_width": pred_w,
            "image_height": pred_h,
            "reference_px": ref_px,
            f"dice_{primary}_native": dice_primary_native,
            f"dice_{secondary}_native": dice_secondary_native,
            f"dice_{primary}_resized": dice_primary_resized,
            f"dice_{secondary}_resized": dice_secondary_resized,
            f"dice_{primary}_interp_baseline": dice_primary_interp,
            f"dice_{secondary}_interp_baseline": dice_secondary_interp,
        })

        print(f"  {px} μm/px — {primary} native: {dice_primary_native:.3f}, "
              f"resized: {dice_primary_resized:.3f}, interp: {dice_primary_interp:.3f}")

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
