#!/usr/bin/env python3
"""
Estimate-then-resample baseline: can the pixel size be estimated from the image itself, well enough
for canonical resampling to work without metadata?

The representation probe (E3) decoded log2(pixel size) from the image's power spectrum to ~0.13 octave,
but it was fit and tested on resampled copies of the same test images. This script asks the question a
practitioner would: train an estimator on some data, apply it to new data, including a different
acquisition.

Features: radially averaged power spectrum of the image, averaged over 96 x 96 tiles (Hann window, mean
removed), normalised to unit total power so that brightness and contrast drop out, log10, 24 radial bins
in cycles per pixel (DC excluded). Image size does not matter, only the pixel-scale content.

Regressors (log2 of the pixel size in nm): ridge on standardised features, and gradient boosting.

Train: TEM1 training subjects (all but the 4 test subjects of subject_split.json), a central 2048 x 2048
native crop of every image resampled exactly as in the sweep (evaluate_nnunet.resample) to 13 pixel
sizes from 1 to 50 nm.
Test:
  tem1_test   the 4 TEM1 test subjects, same crop, all 25 sweep sizes (same microscope and tissue)
  tem2_full   the TEM2 images, acquired natively at 4.93 nm (another acquisition), central crop, resampled
              to the TEM2 sweep sizes 2.36-21.3 nm (4.93 = native, untouched)
  tem2_roi    the 21 annotated TEM2 ROIs (600 x 600) that the paper scores, same sizes
How to read: canonical resampling keeps its accuracy for errors k = estimate / true between ~0.67 and 1.5,
loses 0.04-0.05 mean Dice at k = 2 or 0.5 and fails at k = 10 or 0.1 (paper, Fig. 4b).

Output: {out}/predictions.csv (one row per image and size) and {out}/summary.csv (per set and size:
median k, mean |log2 k|, fraction within 0.67-1.5 and within 0.5-2).

Usage (tassan, CPU, inside a set_slot):
  source ~/resinv_exp/venv_resinv_v2/bin/activate
  python estimate_pixel_size.py --workers 8
"""

import argparse
import csv
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from evaluate_nnunet import resample  # noqa: E402  same resampling as the sweep

Image.MAX_IMAGE_PIXELS = None
HOME = Path.home()

TRAIN_NM = [1.0, 1.52, 2.36, 3.48, 4.93, 7.0, 10.0, 13.93, 16.0, 21.3, 28.3, 37.6, 50.0]
TEM1_TEST_NM = [1.0, 1.15, 1.32, 1.52, 1.74, 2.0, 2.36, 2.64, 3.03, 3.48, 4.0, 4.59, 4.93, 6.06, 7.0,
                8.0, 9.19, 10.0, 12.13, 13.93, 16.0, 21.3, 28.3, 37.6, 50.0]
TEM2_NM = [2.36, 3.48, 4.93, 7.0, 10.0, 16.0, 21.3]
TEM1_NM, TEM2_NATIVE_NM = 2.36, 4.93
CROP, TILE, N_BINS, MAX_TILES = 2048, 96, 24, 64


def spectrum(img: np.ndarray) -> np.ndarray:
    """Log10 of the normalised, radially averaged power spectrum over TILE x TILE tiles."""
    x = img.astype(np.float32)
    if x.shape[0] < TILE or x.shape[1] < TILE:  # tiny coarse images: reflect up to one tile
        x = np.pad(x, ((0, max(0, TILE - x.shape[0])), (0, max(0, TILE - x.shape[1]))), mode="reflect")
    ys = range(0, x.shape[0] - TILE + 1, max(TILE // 2, (x.shape[0] - TILE) // 7 or 1))
    xs = range(0, x.shape[1] - TILE + 1, max(TILE // 2, (x.shape[1] - TILE) // 7 or 1))
    pos = [(y, z) for y in ys for z in xs][:MAX_TILES]
    win = np.outer(np.hanning(TILE), np.hanning(TILE)).astype(np.float32)
    f = np.fft.fftfreq(TILE)
    r = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    bins = np.minimum((r / 0.5 * N_BINS).astype(int), N_BINS)  # bin N_BINS = corners beyond Nyquist
    power = np.zeros(N_BINS + 1)
    for y, z in pos:
        t = x[y:y + TILE, z:z + TILE]
        p = np.abs(np.fft.fft2((t - t.mean()) * win)) ** 2
        power += np.bincount(bins.ravel(), weights=p.ravel(), minlength=N_BINS + 1)
    counts = np.bincount(bins.ravel(), minlength=N_BINS + 1)
    radial = (power / np.maximum(counts, 1))[1:N_BINS]  # drop DC bin and corners
    radial = radial / radial.sum()
    return np.log10(radial + 1e-12)


def central_crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape
    y0, x0 = max(0, (h - size) // 2), max(0, (w - size) // 2)
    return img[y0:y0 + size, x0:x0 + size]


def process(job):
    """job = (set, path, native_nm, sizes, crop) -> rows of features."""
    name, path, native_nm, sizes, crop = job
    img = np.array(Image.open(path).convert("L"))
    if crop:
        img = central_crop(img, crop)
    rows = []
    for nm in sizes:
        scale = native_nm / nm
        im = img if abs(scale - 1) < 1e-4 else resample(img, scale)
        rows.append({"set": name, "image": Path(path).stem, "subject": Path(path).stem.split("_")[0],
                     "true_nm": nm, "feat": spectrum(im)})
    return rows


def run(jobs, workers):
    out, t0 = [], time.time()
    with ProcessPoolExecutor(workers) as ex:
        for i, rows in enumerate(ex.map(process, jobs, chunksize=1), 1):
            out += rows
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} images, {time.time() - t0:.0f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tem1-dir", type=Path, default=HOME / "resinv_exp/data/TEM1")
    ap.add_argument("--tem2-dir", type=Path, default=HOME / "resinv_exp/data/TEM2/001350")
    ap.add_argument("--tem2-roi-dir", type=Path, default=HOME / "resinv_exp/data/testset_armand_uaxon")
    ap.add_argument("--out-dir", type=Path, default=HOME / "resinv_exp/results_paper/px_estimation")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-images", type=int, default=None, help="per set, for a quick test")
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    test_subjects = set(json.loads((args.tem1_dir / "subject_split.json").read_text())["test_subjects"])
    tem1 = sorted(p for p in glob.glob(str(args.tem1_dir / "sub-*/micr/*.png")) if "_seg-" not in p)
    train = [p for p in tem1 if Path(p).stem.split("_")[0] not in test_subjects]
    test = [p for p in tem1 if Path(p).stem.split("_")[0] in test_subjects]
    tem2 = sorted(p for p in glob.glob(str(args.tem2_dir / "sub-*/micr/*.png")) if "_seg-" not in p)
    roi = sorted(glob.glob(str(args.tem2_roi_dir / "sub-*/micr/*acq-roi_TEM.png")))
    cap = (lambda L: L[:args.max_images]) if args.max_images else (lambda L: L)
    print(f"TEM1 train {len(train)} images, TEM1 test {len(test)} ({sorted(test_subjects)}), "
          f"TEM2 {len(tem2)}, TEM2 ROIs {len(roi)}", flush=True)

    print("Features: train", flush=True)
    tr = run([("train", p, TEM1_NM, TRAIN_NM, CROP) for p in cap(train)], args.workers)
    print("Features: test sets", flush=True)
    te = run([("tem1_test", p, TEM1_NM, TEM1_TEST_NM, CROP) for p in cap(test)]
             + [("tem2_full", p, TEM2_NATIVE_NM, TEM2_NM, CROP) for p in cap(tem2)]
             + [("tem2_roi", p, TEM2_NATIVE_NM, TEM2_NM, None) for p in cap(roi)], args.workers)

    X = np.stack([r["feat"] for r in tr]); y = np.log2([r["true_nm"] for r in tr])
    models = {
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 15))).fit(X, y),
        "hgb": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(X, y),
    }
    Xt = np.stack([r["feat"] for r in te])
    est = {k: 2 ** m.predict(Xt) for k, m in models.items()}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "image", "subject", "true_nm", "est_ridge_nm", "est_hgb_nm"])
        for i, r in enumerate(te):
            w.writerow([r["set"], r["image"], r["subject"], r["true_nm"],
                        round(float(est["ridge"][i]), 4), round(float(est["hgb"][i]), 4)])

    summary = []
    sets = np.array([r["set"] for r in te]); true = np.array([r["true_nm"] for r in te])
    print(f"\n{'set':10s} {'true nm':>8s} {'model':6s} {'median k':>9s} {'|log2 k|':>9s} "
          f"{'in .67-1.5':>11s} {'in .5-2':>8s}  n")
    for s in ("tem1_test", "tem2_full", "tem2_roi"):
        for nm in sorted(set(true[sets == s])):
            sel = (sets == s) & (true == nm)
            for m in models:
                k = est[m][sel] / nm
                row = {"set": s, "true_nm": nm, "model": m, "n": int(sel.sum()),
                       "median_k": round(float(np.median(k)), 3),
                       "mean_abs_log2_k": round(float(np.mean(np.abs(np.log2(k)))), 3),
                       "frac_within_0.67_1.5": round(float(np.mean((k >= 0.67) & (k <= 1.5))), 3),
                       "frac_within_0.5_2": round(float(np.mean((k >= 0.5) & (k <= 2))), 3)}
                summary.append(row)
                if s != "tem1_test" or nm in (1.0, 2.36, 4.93, 10.0, 21.3, 50.0):
                    print(f"{s:10s} {nm:8g} {m:6s} {row['median_k']:9.3f} {row['mean_abs_log2_k']:9.3f} "
                          f"{row['frac_within_0.67_1.5']:11.2f} {row['frac_within_0.5_2']:8.2f}  {row['n']}")
    with open(args.out_dir / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader(); w.writerows(summary)
    print(f"\n-> {args.out_dir}/predictions.csv, summary.csv")


if __name__ == "__main__":
    main()
