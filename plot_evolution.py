#!/usr/bin/env python3
"""
Plot resolution evolution: show the same physical crop at each tested resolution,
with segmentation overlays, for one representative image.

Usage:
    python plot_evolution.py --results-dir ./results_tem1 --data-dir /path/to/TEM1
    python plot_evolution.py --results-dir ./results_tem1 --data-dir /path/to/TEM1 \
        --image sub-nyuMouse07_sample-0001_TEM --crop-um 3.0
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.transform import resize

Image.MAX_IMAGE_PIXELS = None

# Pixel sizes to show (one column each). Must exist in the downsampled folder.
DISPLAY_PX = [0.00236, 0.003261, 0.00493, 0.0068833, 0.01, 0.016]
PX_LABELS   = ["2.4 nm", "3.3 nm", "4.9 nm\n(train)", "6.9 nm", "10 nm", "16 nm"]

# Overlay colors (RGBA, 0-1)
AXON_COLOR   = np.array([1.0, 0.85, 0.0, 0.45])   # yellow
MYELIN_COLOR = np.array([0.0, 0.55, 1.0, 0.45])   # blue
UAXON_COLOR  = np.array([1.0, 0.35, 0.0, 0.45])   # orange


def load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


def parse_px(filename: str) -> float | None:
    m = re.search(r"_px([\d.]+)um", filename)
    return float(m.group(1)) if m else None


def find_file(directory: Path, px: float, suffix: str = "") -> Path | None:
    """Find a file in directory whose name contains _px{px}um{suffix}."""
    for p in directory.iterdir():
        fpx = parse_px(p.name)
        if fpx is not None and abs(fpx - px) < 1e-6:
            if suffix == "" or p.name.endswith(suffix):
                return p
    return None


def center_crop_px(arr: np.ndarray, h_px: int, w_px: int) -> np.ndarray:
    """Crop a centered h_px × w_px region from arr."""
    cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
    y0 = max(0, cy - h_px // 2)
    x0 = max(0, cx - w_px // 2)
    y1 = min(arr.shape[0], y0 + h_px)
    x1 = min(arr.shape[1], x0 + w_px)
    return arr[y0:y1, x0:x1]


def to_display(crop: np.ndarray, display_size: int) -> np.ndarray:
    """Resize crop to display_size × display_size using nearest-neighbor (preserves pixelation)."""
    out = resize(crop, (display_size, display_size), order=0,
                 preserve_range=True, anti_aliasing=False)
    return out.astype(np.uint8)


def overlay_masks(gray: np.ndarray, masks: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Composite colored mask overlays onto a grayscale image.
    masks: list of (binary_mask, rgba_color) pairs.
    Returns an RGB uint8 array.
    """
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    for mask, color in masks:
        alpha = (mask > 0).astype(np.float32) * color[3]
        for c in range(3):
            rgb[:, :, c] = rgb[:, :, c] * (1 - alpha) + color[c] * 255 * alpha
    return np.clip(rgb, 0, 255).astype(np.uint8)


def pick_image(results_dir: Path, model_name: str) -> str:
    """Pick the first image directory that has downsampled files."""
    model_dir = results_dir / model_name
    for d in sorted(model_dir.iterdir()):
        if d.is_dir() and any(d.iterdir()):
            return d.name
    raise RuntimeError(f"No image directories found in {model_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("./results_tem1"))
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Dataset root for GT masks (optional)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (default: first found)")
    parser.add_argument("--image", type=str, default=None,
                        help="Image stem to use (default: first available)")
    parser.add_argument("--crop-um", type=float, default=0.8,
                        help="Physical crop size in μm (default: 0.8)")
    parser.add_argument("--display-size", type=int, default=320,
                        help="Display pixels per panel (default: 320)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    # Resolve model
    model_name = args.model
    if model_name is None:
        model_name = next(d.name for d in sorted(args.results_dir.iterdir()) if d.is_dir())

    model_dir = args.results_dir / model_name

    # Resolve image
    img_name = args.image or pick_image(args.results_dir, model_name)
    img_dir = model_dir / img_name
    ds_dir = img_dir / "downsampled"
    pred_dir = img_dir / "predictions"

    print(f"Model: {model_name}")
    print(f"Image: {img_name}")
    print(f"Crop:  {args.crop_um} μm × {args.crop_um} μm")

    # Detect available labels from predictions at finest resolution
    finest_px = DISPLAY_PX[0]
    all_pred = list(pred_dir.glob("*_seg-*.png"))
    labels_present = sorted({
        re.search(r"_seg-(.+)\.png", p.name).group(1)
        for p in all_pred if re.search(r"_seg-(.+)\.png", p.name)
    })
    # Keep only the interesting ones, in display order
    label_order = ["axon", "myelin", "uaxon"]
    show_labels = [l for l in label_order if l in labels_present]
    label_colors = {"axon": AXON_COLOR, "myelin": MYELIN_COLOR, "uaxon": UAXON_COLOR}
    label_names  = {"axon": "axon (pred)", "myelin": "myelin (pred)", "uaxon": "uaxon (pred)"}

    n_cols = len(DISPLAY_PX)
    panel_size = 5.0  # inches per panel

    fig, axes = plt.subplots(1, n_cols, figsize=(panel_size * n_cols, panel_size + 0.6))
    fig.suptitle(f"Resolution evolution: {img_name}", fontsize=13, y=1.01)

    D = args.display_size

    for col_idx, (px, px_label) in enumerate(zip(DISPLAY_PX, PX_LABELS)):
        ax = axes[col_idx]

        raw_path = find_file(ds_dir, px)
        if raw_path is None:
            print(f"  Missing raw at {px} μm/px, skipping column")
            ax.axis("off")
            continue

        raw = load_gray(raw_path)
        h_px = max(1, round(args.crop_um / px))
        w_px = max(1, round(args.crop_um / px))

        raw_crop = center_crop_px(raw, h_px, w_px)
        raw_disp = to_display(raw_crop, D)

        ax.imshow(raw_disp, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(px_label, fontsize=11)
        print(f"  {px} μm/px — crop {w_px}×{h_px} px → displayed at {D}×{D}")

    plt.tight_layout()
    out_path = args.out or (model_dir / f"evolution_{img_name}.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
