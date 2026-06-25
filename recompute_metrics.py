#!/usr/bin/env python3
"""
Recompute Dice metrics from existing prediction PNGs using MONAI.
Optimized: parallel image processing with multiprocessing Pool.

For axon/myelin labels: Dice is computed against manual GT masks from
  {data_dir}/derivatives/labels/{subject}/micr/{img_name}_seg-{label}-manual.png

For uaxon (no GT available): Dice is computed against the model prediction
  at the closest pixel size to training resolution (0.00493 μm/px).

Usage:
    python recompute_metrics.py --results-dir ./results --data-dir ./data/TEM1
    python recompute_metrics.py --results-dir ./results   # uaxon-only mode
    python recompute_metrics.py --results-dir ./results --data-dir ./data/TEM2 --workers 32
"""

import argparse
import re
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.metrics import compute_dice
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

TRAINING_PX = 0.00493
ORIGINAL_PX = 0.0018625
GT_LABELS = {"axon", "myelin"}
SKIP_LABELS = {"nuclei", "process", "axonmyelin"}


def load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


def resample_mask(img: np.ndarray, target_shape: tuple) -> np.ndarray:
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


def load_gt_mask(img_name: str, label: str, data_dir: Path) -> np.ndarray | None:
    """Load GT mask for a label. Falls back to splitting axonmyelin if separate mask missing."""
    subject = img_name.split("_")[0]
    base = data_dir / "derivatives" / "labels" / subject / "micr"

    direct = base / f"{img_name}_seg-{label}-manual.png"
    if direct.exists():
        return load_gray(direct)

    # Split axonmyelin: white (255) = axon, gray (>0 and <255) = myelin
    combined = base / f"{img_name}_seg-axonmyelin-manual.png"
    if combined.exists():
        arr = load_gray(combined)
        if label == "axon":
            return (arr == 255).astype(np.uint8) * 255
        elif label == "myelin":
            return ((arr > 0) & (arr < 255)).astype(np.uint8) * 255

    return None


def detect_active_labels(pred_dir: Path, ref_px: float) -> list[str]:
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
        if 0 < (arr > 0).mean() < 0.5:
            active.append(label)
    return active


def recompute_image(img_dir: Path, model_name: str, data_dir: Path | None, gt_only: bool = False) -> pd.DataFrame:
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
        return pd.DataFrame()

    # Load reference masks (GT if available, else native-px prediction)
    ref_masks = {}
    ref_sources = {}
    for label in active_labels:
        gt_arr = load_gt_mask(img_name, label, data_dir) if (data_dir and label in GT_LABELS) else None
        if gt_arr is not None:
            ref_masks[label] = gt_arr
            ref_sources[label] = "gt"
        else:
            candidates = [p for p in pred_dir.glob(f"*_seg-{label}.png")
                          if parse_px(p.name) is not None and abs(parse_px(p.name) - ref_px) < 1e-7]
            if candidates:
                ref_masks[label] = load_gray(candidates[0])
                ref_sources[label] = "pred"

    if not ref_masks:
        return pd.DataFrame()

    if gt_only and all(v == "pred" for v in ref_sources.values()):
        print(f"  [{img_dir.name}] skipping — no GT found", flush=True)
        return pd.DataFrame()

    ref_shapes = {label: ref_masks[label].shape for label in ref_masks}

    # Pre-load all prediction files grouped by px to avoid redundant globs
    preds_by_px: dict[float, dict[str, Path]] = {}
    for p in all_pred_files:
        px = parse_px(p.name)
        m = re.search(r"_seg-(.+)\.png", p.name)
        if px is None or m is None:
            continue
        label = m.group(1)
        preds_by_px.setdefault(px, {})[label] = p

    rows = []
    for px in px_set:
        label_files = preds_by_px.get(px, {})
        if not label_files:
            continue

        sample_path = next(iter(label_files.values()))
        pred_h, pred_w = load_gray(sample_path).shape

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
            if label not in ref_masks or label not in label_files:
                continue
            ref_mask = ref_masks[label]
            ref_h, ref_w = ref_shapes[label]
            pred = load_gray(label_files[label])

            pred_at_ref = resample_mask(pred, (ref_h, ref_w))
            ref_at_pred = resample_mask(ref_mask, (pred_h, pred_w))

            row[f"dice_{label}_native"] = dice_monai(pred_at_ref, ref_mask)
            row[f"dice_{label}_resized"] = dice_monai(pred, ref_at_pred)
            row[f"dice_{label}_interp_baseline"] = dice_monai(
                resample_mask(ref_at_pred, (ref_h, ref_w)), ref_mask
            )

        rows.append(row)
        label_summary = ", ".join(
            f"{l}({ref_sources.get(l, '?')}) {row.get(f'dice_{l}_native', float('nan')):.3f}"
            for l in active_labels if l in ref_masks
        )
        print(f"  [{img_name}] {px:.7g} μm/px — {label_summary}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(img_dir / "metrics.csv", index=False)
    return df


def _worker(args: tuple) -> pd.DataFrame:
    img_dir, model_name, data_dir, gt_only = args
    try:
        return recompute_image(img_dir, model_name, data_dir, gt_only)
    except Exception as e:
        print(f"  ERROR [{args[0].name}]: {e}", flush=True)
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("./results"))
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Dataset root (BIDS). Needed to load GT masks for axon/myelin.")
    parser.add_argument("--workers", type=int, default=min(cpu_count(), 32),
                        help="Parallel workers (default: min(cpu_count, 32))")
    parser.add_argument("--gt-only", action="store_true",
                        help="Skip images where no GT mask is found (all labels fall back to pred)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Only process these model names (e.g. --models da5 da5_multires)")
    args = parser.parse_args()

    for model_dir in sorted(args.results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        if args.models and model_name not in args.models:
            continue
        print(f"\n{'='*60}\nModel: {model_name} — {args.workers} workers")

        img_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir()])
        worker_args = [(d, model_name, args.data_dir, args.gt_only) for d in img_dirs]

        with Pool(args.workers) as pool:
            dfs = pool.map(_worker, worker_args)

        all_dfs = [df for df in dfs if not df.empty]
        if all_dfs:
            summary = pd.concat(all_dfs, ignore_index=True)
            summary.to_csv(model_dir / "results.csv", index=False)
            print(f"\nSaved: {model_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
