#!/usr/bin/env python3
"""
Linear probe for pixel size on nnU-Net features, from extract_features.py output (E3).

Ridge regression (standardized features, alpha by internal CV) predicts log2(pixel size) from
each stage's features, cross-validated leave-one-subject-out over the TEM1 test mice, so no mouse
is on both sides. Two levels:
  global  one vector per image and pixel size (mean over the image)
  obj     one vector per GT object (axons and sheaths, fully inside the processed region)
Reports R2 and mean absolute error in octaves (|log2 predicted - log2 true|).

Baseline (--data-dir): the same probe on the radially averaged log power spectrum of the network
input itself, in cycles per pixel. A stage that decodes pixel size better than this baseline
represents it beyond what blur and texture in the image already give away.

Caveat: from 8 nm on, TEM1 images are smaller than the patch and get zero-padded, and InstanceNorm
statistics include the padding. Pixel size can leak through that. --px-max 0.0213 restricts the
probe to sizes that matter for the robust band; run it both ways.

Usage (tassan, CPU):
  python probe.py --feat-dir ~/resinv_exp/results_paper/representation \\
      --models control_f0 multires_f0 multires2_f0 multires_gap10_f0 \\
      --data-dir ~/resinv_exp/data/TEM1 \\
      --out ~/resinv_exp/results_paper/representation/probe.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cka import Features

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from evaluate_nnunet import resample  # noqa: E402  same resampling as the sweep

Image.MAX_IMAGE_PIXELS = None
ALPHAS = np.logspace(-3, 5, 17)


def probe(x: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """LOSO predictions of y from x. Returns (r2, mae in octaves, predictions)."""
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    pred = cross_val_predict(model, x, y, groups=groups, cv=LeaveOneGroupOut())
    r2 = 1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return float(r2), float(np.abs(pred - y).mean()), pred


def power_spectrum(img: np.ndarray, n_bins: int = 32) -> np.ndarray:
    """Radially averaged log power spectrum, frequency in cycles/pixel (0 to 0.5)."""
    f = np.abs(np.fft.fftshift(np.fft.fft2(img.astype(np.float32) - img.mean()))) ** 2
    h, w = img.shape
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2)
    bins = np.minimum((r / 0.5 * n_bins).astype(int), n_bins)
    p = np.bincount(bins.ravel(), weights=f.ravel(), minlength=n_bins + 1)[:n_bins]
    n = np.bincount(bins.ravel(), minlength=n_bins + 1)[:n_bins]
    return np.log10(p / np.maximum(n, 1) + 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--feat-dir", type=Path, required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--data-dir", type=Path, default=None, help="TEM1 root, for the power-spectrum baseline")
    ap.add_argument("--original-px", type=float, default=0.00236)
    ap.add_argument("--px-min", type=float, default=0.0)
    ap.add_argument("--px-max", type=float, default=1.0)
    ap.add_argument("--levels", nargs="+", default=["global", "obj"])
    ap.add_argument("--max-obj-per-size", type=int, default=3000, help="random object cap per pixel size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows, preds = [], []

    def record(model, stage, level, x, y, groups, px):
        r2, mae, pred = probe(x, y, groups)
        rows.append({"model": model, "stage": stage, "level": level, "n": len(y),
                     "px_min_nm": round(px.min() * 1000, 3), "px_max_nm": round(px.max() * 1000, 3),
                     "r2": round(r2, 4), "mae_octaves": round(mae, 4)})
        for p in np.unique(px):
            k = px == p
            preds.append({"model": model, "stage": stage, "level": level, "px_nm": round(p * 1000, 3),
                          "pred_nm_median": round(float(2 ** np.median(pred[k])) * 1000, 3),
                          "mae_octaves": round(float(np.abs(pred[k] - y[k]).mean()), 4)})
        print(f"  {model:18s} {stage:5s} {level:6s} n={len(y):6d}  R2 {r2:6.3f}  MAE {mae:.3f} oct", flush=True)

    for m in args.models:
        F = Features(args.feat_dir / m, "obj")
        sel = np.flatnonzero((F.px >= args.px_min - 1e-9) & (F.px <= args.px_max + 1e-9))
        px_sel = F.px[sel]
        img_subject = np.array([Path(f).name.split("_")[0] for f in F.files])
        for s in F.stages:
            if "global" in args.levels:
                g = np.stack([np.load(f)[f"global_{s}"][sel] for f in F.files])  # (I, P, C)
                x = g.reshape(-1, g.shape[-1])
                px = np.tile(px_sel, len(F.files))
                record(m, s, "global", x, np.log2(px), np.repeat(img_subject, len(sel)), px)
            if "obj" in args.levels:
                f, _ = F.stage(s)
                xs, ys, gs, ps = [], [], [], []
                for i in sel:
                    idx = np.flatnonzero((F.frac[i] == 1) & ~np.isnan(f[i]).any(1))
                    if len(idx) > args.max_obj_per_size:
                        idx = rng.choice(idx, args.max_obj_per_size, replace=False)
                    xs.append(f[i, idx]); gs.append(F.subject[idx])
                    ys.append(np.full(len(idx), np.log2(F.px[i]))); ps.append(np.full(len(idx), F.px[i]))
                record(m, s, "obj", np.concatenate(xs), np.concatenate(ys), np.concatenate(gs), np.concatenate(ps))

    if args.data_dir:
        F = Features(args.feat_dir / args.models[0], "obj")
        sel = F.px[(F.px >= args.px_min - 1e-9) & (F.px <= args.px_max + 1e-9)]
        x, y, g = [], [], []
        for f in F.files:
            name = Path(f).stem
            subject = name.split("_")[0]
            img = np.array(Image.open(args.data_dir / subject / "micr" / f"{name}.png").convert("L"))
            for p in sel:
                s = args.original_px / p
                x.append(power_spectrum(img if abs(s - 1) < 1e-4 else resample(img, s)))
                y.append(np.log2(p)); g.append(subject)
        record("baseline", "spectrum", "global", np.array(x), np.array(y), np.array(g), np.tile(sel, len(F.files)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for path, data in ((args.out, rows), (args.out.with_name(args.out.stem + "_per_size.csv"), preds)):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
