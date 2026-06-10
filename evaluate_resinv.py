#!/usr/bin/env python3
"""
Resolution invariance evaluation for ADS segmentation models.

Takes the original 0.0018625 μm/px Corpus Callosum images, downsamples them to
a range of pixel sizes, runs ADS segmentation on each, then computes Dice
consistency against the prediction at training resolution (0.00493 μm/px).

Handles two evaluation axes:
  - native space:  all predictions resized back to reference (0.00493) → Dice
  - resized space: prediction vs reference both at the downsampled resolution → Dice

Also computes an interpolation baseline (round-trip resize of the reference
prediction at each scale) to separate model degradation from resampling artifacts.

Usage:
    python evaluate_resinv.py \\
        --data-dir /path/to/Corpus_Callosum \\
        --model-path /path/to/model \\
        --model-name cambridge_unmyelinated \\
        --output-dir ./results \\
        --gpu-id 0
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import resize

# Disable PIL decompression bomb check (original TEM images are ~230M pixels)
Image.MAX_IMAGE_PIXELS = None

TRAINING_PX = 0.00493  # μm/px — the model's native resolution
MIN_IMAGE_SIZE = 128   # skip resolutions where min(H, W) < this

# 10 log-spaced points from full-res to 0.01, then 6 more up to 0.016
# Training res 0.00493 is injected explicitly so the reference prediction is exact
_low = np.logspace(np.log10(0.0018625), np.log10(0.01), 10)
_high = np.logspace(np.log10(0.01), np.log10(0.016), 7)[1:]  # skip 0.01, already in _low
_all = np.round(np.concatenate([_low, _high]), 7)
_all = np.sort(np.unique(np.append(_all, TRAINING_PX)))
DEFAULT_PIXEL_SIZES: list[float] = list(_all)


# ---------------------------------------------------------------------------
# Image I/O and resampling
# ---------------------------------------------------------------------------

def load_gray(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.array(img)


def resample(img: np.ndarray, target_shape: tuple[int, int], is_mask: bool = False) -> np.ndarray:
    """Resize to target_shape (H, W). Nearest-neighbor for masks, bicubic for images."""
    order = 0 if is_mask else 3
    anti_alias = (not is_mask) and (target_shape[0] < img.shape[0])
    out = resize(img, target_shape, order=order,
                 preserve_range=True, anti_aliasing=anti_alias)
    return out.astype(img.dtype)


def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    if img.max() == 0:
        return img.astype(np.uint8)
    return ((img.astype(np.float32) / img.max()) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def dice(a: np.ndarray, b: np.ndarray) -> float:
    a_bin = a > 0
    b_bin = b > 0
    intersection = (a_bin & b_bin).sum()
    total = a_bin.sum() + b_bin.sum()
    if total == 0:
        return 1.0
    return float(2 * intersection / total)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def find_original_images(data_dir: Path) -> list[Path]:
    """Find original TIF images (not pre-resampled versions, not masks)."""
    all_tifs = list(data_dir.rglob("*.tif"))
    return sorted(
        p for p in all_tifs
        if "px0.00493um" not in p.name
        and "patches" not in str(p)
        and "_seg-" not in p.name
    )


def px_tag(px: float) -> str:
    return f"px{px:.6g}um"


# ---------------------------------------------------------------------------
# Per-image evaluation
# ---------------------------------------------------------------------------

def evaluate_image(
    image_path: Path,
    model_path: Path,
    model_name: str,
    original_px: float,
    pixel_sizes: list[float],
    output_dir: Path,
    gpu_id: int,
    primary_label: str = "axon",
    secondary_label: str = "myelin",
    crop_size: int | None = None,
) -> pd.DataFrame:

    img_name = image_path.stem
    img_out = output_dir / model_name / img_name
    ds_dir = img_out / "downsampled"
    pred_dir = img_out / "predictions"
    ds_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Image: {img_name}")

    img = load_gray(image_path)
    orig_h, orig_w = img.shape

    if crop_size is not None and (orig_h > crop_size or orig_w > crop_size):
        cy, cx = orig_h // 2, orig_w // 2
        half = crop_size // 2
        img = img[cy - half:cy + half, cx - half:cx + half]
        orig_h, orig_w = img.shape
        print(f"  Cropped to: {orig_w}×{orig_h} px (centered)")

    print(f"  Original: {orig_w}×{orig_h} px  @ {original_px} μm/px")

    # --- Determine valid pixel sizes (downsampling only, image large enough) ---
    valid: list[tuple[float, Path]] = []  # (pixel_size, downsampled_path)

    for px in sorted(pixel_sizes):
        if px < original_px - 1e-9:
            print(f"  Skip {px} μm/px: upsampling not tested")
            continue
        scale = original_px / px
        new_h = max(1, round(orig_h * scale))
        new_w = max(1, round(orig_w * scale))
        if min(new_h, new_w) < MIN_IMAGE_SIZE:
            print(f"  Skip {px} μm/px: result too small ({new_w}×{new_h})")
            continue

        tag = px_tag(px)
        out_path = ds_dir / f"{img_name}_{tag}.png"
        if not out_path.exists():
            ds_img = resample(img, (new_h, new_w), is_mask=False)
            Image.fromarray(to_uint8(ds_img)).save(out_path)
        valid.append((px, out_path))
        print(f"  {px} μm/px → {new_w}×{new_h} px")

    if not valid:
        print("  No valid pixel sizes, skipping.")
        return pd.DataFrame()

    # --- Run ADS on images that don't have predictions yet ---
    to_segment = []
    for px, ds_path in valid:
        tag = px_tag(px)
        in_pred_dir = pred_dir / f"{img_name}_{tag}_seg-axon.png"
        in_ds_dir = ds_path.parent / f"{ds_path.stem}_seg-axon.png"
        if not in_pred_dir.exists() and not in_ds_dir.exists():
            to_segment.append(ds_path)

    if to_segment:
        print(f"  Segmenting {len(to_segment)} image(s)...")
        from AxonDeepSeg.segment import segment_images
        segment_images(
            path_images=to_segment,
            path_model=model_path,
            gpu_id=gpu_id,
            allow_large_images=True,
        )
        for ds_path in to_segment:
            for src in ds_path.parent.glob(f"{ds_path.stem}_seg-*.png"):
                shutil.move(str(src), pred_dir / src.name)
    else:
        print("  All predictions already exist.")

    # --- Find reference prediction (closest to training resolution) ---
    closest_px = min(valid, key=lambda t: abs(t[0] - TRAINING_PX))[0]
    ref_tag = px_tag(closest_px)
    ref_primary_path = pred_dir / f"{img_name}_{ref_tag}_seg-{primary_label}.png"
    ref_secondary_path = pred_dir / f"{img_name}_{ref_tag}_seg-{secondary_label}.png"

    if not ref_primary_path.exists():
        print(f"  Reference prediction missing ({ref_primary_path.name}), skipping metrics.")
        return pd.DataFrame()

    ref_primary = load_gray(ref_primary_path)
    ref_secondary = load_gray(ref_secondary_path) if ref_secondary_path.exists() else None
    ref_h, ref_w = ref_primary.shape

    # --- Compute metrics for each resolution ---
    rows = []
    for px, ds_path in valid:
        tag = px_tag(px)
        scale = original_px / px
        pred_w = max(1, round(orig_w * scale))
        pred_h = max(1, round(orig_h * scale))

        pred_primary_path = pred_dir / f"{img_name}_{tag}_seg-{primary_label}.png"
        pred_secondary_path = pred_dir / f"{img_name}_{tag}_seg-{secondary_label}.png"

        if not pred_primary_path.exists():
            print(f"  Missing prediction at {px} μm/px, skipping.")
            continue

        pred_primary = load_gray(pred_primary_path)
        pred_secondary = load_gray(pred_secondary_path) if pred_secondary_path.exists() else None

        # Native space: resize prediction to reference space, compute Dice vs reference
        pred_primary_native = resample(pred_primary, (ref_h, ref_w), is_mask=True)
        dice_primary_native = dice(pred_primary_native, ref_primary)

        dice_secondary_native = None
        if pred_secondary is not None and ref_secondary is not None:
            pred_secondary_native = resample(pred_secondary, (ref_h, ref_w), is_mask=True)
            dice_secondary_native = dice(pred_secondary_native, ref_secondary)

        # Resized space: resize reference to prediction space, compute Dice directly
        ref_primary_ds = resample(ref_primary, (pred_h, pred_w), is_mask=True)
        dice_primary_resized = dice(pred_primary, ref_primary_ds)

        dice_secondary_resized = None
        if pred_secondary is not None and ref_secondary is not None:
            ref_secondary_ds = resample(ref_secondary, (pred_h, pred_w), is_mask=True)
            dice_secondary_resized = dice(pred_secondary, ref_secondary_ds)

        # Interpolation baseline: round-trip resize of reference
        ref_primary_roundtrip = resample(
            resample(ref_primary, (pred_h, pred_w), is_mask=True),
            (ref_h, ref_w), is_mask=True
        )
        dice_primary_interp = dice(ref_primary_roundtrip, ref_primary)

        dice_secondary_interp = None
        if ref_secondary is not None:
            ref_secondary_roundtrip = resample(
                resample(ref_secondary, (pred_h, pred_w), is_mask=True),
                (ref_h, ref_w), is_mask=True
            )
            dice_secondary_interp = dice(ref_secondary_roundtrip, ref_secondary)

        row = {
            "image": img_name,
            "model": model_name,
            "pixel_size_um": px,
            "scale_factor": scale,
            "image_width": pred_w,
            "image_height": pred_h,
            "reference_px": closest_px,
            f"dice_{primary_label}_native": dice_primary_native,
            f"dice_{secondary_label}_native": dice_secondary_native,
            f"dice_{primary_label}_resized": dice_primary_resized,
            f"dice_{secondary_label}_resized": dice_secondary_resized,
            f"dice_{primary_label}_interp_baseline": dice_primary_interp,
            f"dice_{secondary_label}_interp_baseline": dice_secondary_interp,
        }
        rows.append(row)
        print(
            f"  {px} μm/px — "
            f"{primary_label} native: {dice_primary_native:.3f}, "
            f"resized: {dice_primary_resized:.3f}, "
            f"interp: {dice_primary_interp:.3f}"
        )

    df = pd.DataFrame(rows)
    per_image_csv = img_out / "metrics.csv"
    df.to_csv(per_image_csv, index=False)
    print(f"  Saved: {per_image_csv}")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global MIN_IMAGE_SIZE
    parser = argparse.ArgumentParser(description="Resolution invariance evaluation for ADS")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Path to Corpus_Callosum dataset directory")
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Path to ADS model directory")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Short name for the model (used in output filenames)")
    parser.add_argument("--original-px", type=float, default=0.0018625,
                        help="Pixel size of the source images in μm/px (default: 0.0018625)")
    parser.add_argument("--pixel-sizes", type=float, nargs="+",
                        default=DEFAULT_PIXEL_SIZES,
                        help="Target pixel sizes to test in μm/px")
    parser.add_argument("--output-dir", type=Path, default=Path("./results"),
                        help="Root output directory (default: ./results)")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="GPU ID to use (-1 for CPU, default: 0)")
    parser.add_argument("--min-size", type=int, default=MIN_IMAGE_SIZE,
                        help="Minimum image dimension in pixels (default: 128)")
    parser.add_argument("--crop-size", type=int, default=None,
                        help="Crop a centered square patch of this size (pixels) from each "
                             "image before processing. Useful for very large images (default: no crop)")
    parser.add_argument("--label", type=str, default="axon",
                        help="Primary segmentation label to evaluate. Use 'uaxon' for the "
                             "unmyelinated TEM model, 'axon' for the generalist (default: axon)")
    parser.add_argument("--secondary-label", type=str, default="myelin",
                        help="Secondary label to evaluate alongside the primary (default: myelin)")
    args = parser.parse_args()
    MIN_IMAGE_SIZE = args.min_size

    images = find_original_images(args.data_dir)
    if not images:
        print(f"No original TIF images found in {args.data_dir}")
        sys.exit(1)

    print(f"Found {len(images)} image(s):")
    for p in images:
        print(f"  {p}")

    all_dfs = []
    for img_path in images:
        df = evaluate_image(
            image_path=img_path,
            model_path=args.model_path,
            model_name=args.model_name,
            original_px=args.original_px,
            pixel_sizes=args.pixel_sizes,
            output_dir=args.output_dir,
            gpu_id=args.gpu_id,
            primary_label=args.label,
            secondary_label=args.secondary_label,
            crop_size=args.crop_size,
        )
        if not df.empty:
            all_dfs.append(df)

    if all_dfs:
        summary = pd.concat(all_dfs, ignore_index=True)
        out_csv = args.output_dir / args.model_name / "results.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_csv, index=False)
        print(f"\nAll results saved to: {out_csv}")
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    main()
