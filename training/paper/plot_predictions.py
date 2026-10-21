#!/usr/bin/env python3
"""
Show the predicted masks of several models across the pixel-size sweep, on the same field of view.

For each test image, one figure: columns are pixel sizes, rows are the test image at that size,
the ground truth, and each model's prediction overlaid on the image. Every panel shows the same
physical crop, drawn at its own pixel grid (nearest neighbour), so coarse panels look blocky as
they really are. Each prediction panel is titled with that image's axon / myelin Dice from the
model's results.csv.

With --errors, a second figure shows what the Dice actually scores: the prediction upsampled with
nearest neighbour to the native ground-truth grid (as recompute_metrics.py does), coloured as
correct, missed (false negative) or extra (false positive) for axon and myelin.

Usage (on tassan, CPU only):
  python training/paper/plot_predictions.py \\
    --models control control_canonical multires --fold 0 \\
    --px 2.36 7 16 28.3 50 --images sub-nyuMouse11_sample-0007_TEM --errors
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import resize

NATIVE_PX = 0.00236  # um/px, TEM1
AXON, MYELIN = np.array([0.16, 0.47, 0.84]), np.array([0.92, 0.41, 0.20])
LABELS = {"control": "Single-resolution", "control_canonical": "Canonical resampling",
          "multires": "Multi-resolution", "multires2": "Ablation: 2.36, 16 nm",
          "multires_to7": "Ablation: 2.36 to 7 nm"}


def px_tag(px_um):
    return f"px{px_um:.7g}um"


def resample(img, scale):
    # Same as evaluate_nnunet.resample, so the panels show exactly what the model was given.
    h, w = max(1, round(img.shape[0] * scale)), max(1, round(img.shape[1] * scale))
    return resize(img, (h, w), order=3, preserve_range=True, anti_aliasing=scale < 1).astype(np.uint8)


def load_mask(path):
    return np.array(Image.open(path).convert("L")) > 0


def load_gt(data_dir, name):
    base = data_dir / "derivatives" / "labels" / name.split("_")[0] / "micr"
    out = {}
    for label in ("axon", "myelin"):
        direct = base / f"{name}_seg-{label}-manual.png"
        if direct.exists():
            out[label] = load_mask(direct)
    if len(out) < 2:
        arr = np.array(Image.open(base / f"{name}_seg-axonmyelin-manual.png").convert("L"))
        out = {"axon": arr == 255, "myelin": (arr > 0) & (arr < 255)}
    return out


def find_image(data_dir, name):
    hits = [p for p in (data_dir / name.split("_")[0] / "micr").glob(f"{name}.*") if "_seg-" not in p.name]
    if not hits:
        raise FileNotFoundError(f"{name} not found under {data_dir}")
    return hits[0]


def overlay(gray, axon, myelin, alpha=0.45):
    rgb = np.repeat(gray[..., None] / 255.0, 3, axis=2)
    for m, c in ((axon, AXON), (myelin, MYELIN)):
        rgb[m] = (1 - alpha) * rgb[m] + alpha * c
    return rgb


def error_map(pred, gt):
    # Background white; correct in the class colour, missed in a dark shade, extra in a light one.
    rgb = np.ones(gt["axon"].shape + (3,))
    for label, c in (("myelin", MYELIN), ("axon", AXON)):
        p, g = pred[label], gt[label]
        rgb[p & g] = c
        rgb[g & ~p] = c * 0.45
        rgb[p & ~g] = 1 - (1 - c) * 0.35
    return rgb


def crop_box(shape, px_um, center, size_um):
    # Crop in pixels at this grid for a physical window centred at `center` (fractions of the image).
    n = max(4, round(size_um / px_um))
    cy, cx = round(center[1] * shape[0]), round(center[0] * shape[1])
    y0 = int(np.clip(cy - n // 2, 0, max(0, shape[0] - n)))
    x0 = int(np.clip(cx - n // 2, 0, max(0, shape[1] - n)))
    return slice(y0, y0 + n), slice(x0, x0 + n)


def dice_of(results, model, name, px_um):
    df = results.get(model)
    if df is None:
        return ""
    r = df[(df.image == name) & (np.isclose(df.pixel_size_um, px_um))]
    if r.empty:
        return ""
    return f"{r.dice_axon_native.iloc[0]:.2f} / {r.dice_myelin_native.iloc[0]:.2f}"


def main():
    home = Path.home() / "resinv_exp"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=home / "results_paper" / "tem1")
    ap.add_argument("--data-dir", type=Path, default=home / "data" / "TEM1")
    ap.add_argument("--models", nargs="+", default=["control", "control_canonical", "multires"])
    ap.add_argument("--fold", default="0")
    ap.add_argument("--px", type=float, nargs="+", default=[2.36, 7, 16, 28.3, 50], help="pixel sizes in nm")
    ap.add_argument("--images", nargs="+", required=True, help="image stems, e.g. sub-nyuMouse11_sample-0007_TEM")
    ap.add_argument("--crop-um", type=float, default=2.5, help="side of the square field of view, um")
    ap.add_argument("--center", type=float, nargs=2, default=[0.5, 0.5], help="crop centre, fractions (x y)")
    ap.add_argument("--errors", action="store_true", help="also plot errors on the native GT grid")
    ap.add_argument("--out-dir", type=Path, default=home / "figures_predictions")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for m in args.models:
        f = args.results_dir / f"{m}_f{args.fold}" / "results.csv"
        if f.exists():
            results[m] = pd.read_csv(f)

    for name in args.images:
        img = np.array(Image.open(find_image(args.data_dir, name)).convert("L"))
        gt = load_gt(args.data_dir, name)
        preds = {}
        for m in args.models:
            d = args.results_dir / f"{m}_f{args.fold}" / name / "predictions"
            preds[m] = {px: {lab: load_mask(d / f"{name}_{px_tag(px / 1000)}_seg-{lab}.png")
                             for lab in ("axon", "myelin")} for px in args.px}

        rows = ["Image", "Ground truth"] + [LABELS.get(m, m) for m in args.models]
        nr, nc = len(rows), len(args.px)
        fig, axes = plt.subplots(nr, nc, figsize=(1.9 * nc, 1.95 * nr), squeeze=False)
        for j, px in enumerate(args.px):
            scale = NATIVE_PX / (px / 1000)
            im_px = img if abs(scale - 1) < 1e-4 else resample(img, scale)
            sy, sx = crop_box(im_px.shape, px / 1000, args.center, args.crop_um)
            gt_px = {k: resize(v, im_px.shape, order=0, preserve_range=True, anti_aliasing=False) > 0
                     for k, v in gt.items()}
            panels = [(im_px[sy, sx] / 255.0, f"{px:g} nm, {im_px.shape[1]}x{im_px.shape[0]} px"),
                      (overlay(im_px[sy, sx], gt_px["axon"][sy, sx], gt_px["myelin"][sy, sx]), "")]
            for m in args.models:
                p = preds[m][px]
                panels.append((overlay(im_px[sy, sx], p["axon"][sy, sx], p["myelin"][sy, sx]),
                               dice_of(results, m, name, px / 1000)))
            for i, (arr, title) in enumerate(panels):
                ax = axes[i, j]
                ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None, vmin=0, vmax=1, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(title, fontsize=7, pad=2)
                if j == 0:
                    ax.set_ylabel(rows[i], fontsize=8)
        fig.suptitle(f"{name}, fold {args.fold}, {args.crop_um:g} um field  (titles: axon / myelin Dice on the full image)",
                     fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = args.out_dir / f"{name}_f{args.fold}_masks.png"
        fig.savefig(out, dpi=180); plt.close(fig)
        print(f"Saved {out}")

        if not args.errors:
            continue
        # What the Dice sees: every prediction mapped to the native GT grid with nearest neighbour.
        sy, sx = crop_box(img.shape, NATIVE_PX, args.center, args.crop_um)
        fig, axes = plt.subplots(len(args.models), nc, figsize=(1.9 * nc, 1.95 * len(args.models)), squeeze=False)
        for j, px in enumerate(args.px):
            for i, m in enumerate(args.models):
                up = {k: resize(v, img.shape, order=0, preserve_range=True, anti_aliasing=False) > 0
                      for k, v in preds[m][px].items()}
                ax = axes[i, j]
                ax.imshow(error_map({k: v[sy, sx] for k, v in up.items()}, {k: v[sy, sx] for k, v in gt.items()}),
                          interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{px:g} nm   {dice_of(results, m, name, px / 1000)}", fontsize=7, pad=2)
                if j == 0:
                    ax.set_ylabel(LABELS.get(m, m), fontsize=8)
        fig.suptitle(f"{name}: prediction on the native GT grid. Full colour = correct, dark = missed, "
                     f"pale = extra (blue axon, orange myelin)", fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out = args.out_dir / f"{name}_f{args.fold}_errors.png"
        fig.savefig(out, dpi=180); plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
