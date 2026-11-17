#!/usr/bin/env python3
"""
Morphometry across the pixel-size sweep: per image and pixel size, mean g-ratio, axon diameter and
myelin thickness of the axons in a segmentation, computed with AxonDeepSeg's own morphometrics
(get_axon_morphometrics) in physical units.

Sources, all on the TEM1 test images:
  gt            manual masks at the native pixel size (the reference, one row per image)
  gt_resampled  manual masks resampled to the test pixel size with nearest neighbour: what is lost
                to pixelation alone (the morphometric counterpart of the resampling ceiling)
  <model>       the model's prediction at the test pixel size (evaluate_nnunet.py output)

Axons touching the image border, and axons without a valid g-ratio (no myelin found), are excluded,
as in AxonDeepSeg's own reports. The per-image g-ratio error is then |mean(pred) - mean(gt)|.

--min-axon-diam-um drops axons with a smaller equivalent diameter, in every source alike. Chosen from
the ground truth alone (2026-11-21): 0.15 um removes 1 of the 1038 manual test axons (smallest 0.095 um,
1st percentile 0.24 um) but the one- to few-pixel specks that make up a third of the objects predicted
at the native size (0.003-0.008 um), and still exceeds a single pixel at 50 nm (0.056 um).

Needs AxonDeepSeg (tassan: ~/resinv_exp/venv_resinv, editable install of ~/axondeepseg). CPU only.

Usage (tassan):
  source ~/resinv_exp/venv_resinv/bin/activate
  python morphometry.py --models control_f0 multires_f0 control_canonical_f0 --workers 2 \\
      --out ~/resinv_exp/results_paper/morphometry_tem1_f0.csv
"""

import argparse
import json
import logging
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import resize

Image.MAX_IMAGE_PIXELS = None
HOME = Path.home()
NATIVE_UM = 0.00236
COLS = ("gratio", "axon_diam", "myelin_thickness")


def load_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 0


def px_tag(px_um: float) -> str:
    return f"px{px_um:.7g}um"


def summarize(axon: np.ndarray, myelin: np.ndarray, px_um: float, min_diam_um: float = 0.0) -> dict:
    """Mean morphometrics of the valid, non-border axons of one segmentation."""
    from AxonDeepSeg.morphometrics.compute_morphometrics import get_axon_morphometrics

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stats = get_axon_morphometrics(axon.astype(np.uint8), im_myelin=myelin.astype(np.uint8),
                                       pixel_size=px_um, return_border_info=True)
    n_all = len(stats)
    if n_all:
        keep = stats["gratio"].notna() & np.isfinite(stats["gratio"])
        if "image_border_touching" in stats:
            keep &= ~stats["image_border_touching"].astype(bool)
        keep &= stats["axon_diam"] >= min_diam_um
        stats = stats[keep]
    out = {"n_axons_all": n_all, "n_axons": len(stats)}
    for c in COLS:
        out[f"{c}_mean"] = float(stats[c].mean()) if len(stats) else np.nan
    return out


def one_image(args):
    img_name, results_dir, data_dir, models, px_list, min_d = args
    subject = img_name.split("_")[0]
    base = data_dir / "derivatives" / "labels" / subject / "micr"
    gt = {lab: load_mask(base / f"{img_name}_seg-{lab}-manual.png") for lab in ("axon", "myelin")}
    gt["myelin"] &= ~gt["axon"]
    rows = [{"image": img_name, "subject": subject, "source": "gt", "px_nm": NATIVE_UM * 1000,
             **summarize(gt["axon"], gt["myelin"], NATIVE_UM, min_d)}]
    shape = gt["axon"].shape
    for px_nm in px_list:
        px_um = px_nm / 1000
        scale = NATIVE_UM / px_um
        size = (max(1, round(shape[0] * scale)), max(1, round(shape[1] * scale)))
        g = {k: resize(v, size, order=0, preserve_range=True, anti_aliasing=False) > 0 for k, v in gt.items()}
        rows.append({"image": img_name, "subject": subject, "source": "gt_resampled", "px_nm": px_nm,
                     **summarize(g["axon"], g["myelin"], px_um, min_d)})
        for m in models:
            d = results_dir / m / img_name / "predictions"
            p = {lab: load_mask(d / f"{img_name}_{px_tag(px_um)}_seg-{lab}.png") for lab in ("axon", "myelin")}
            p["myelin"] &= ~p["axon"]
            rows.append({"image": img_name, "subject": subject, "source": m, "px_nm": px_nm,
                         **summarize(p["axon"], p["myelin"], px_um, min_d)})
    print(f"done {img_name}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results-dir", type=Path, default=HOME / "resinv_exp/results_paper/tem1")
    ap.add_argument("--data-dir", type=Path, default=HOME / "resinv_exp/data/TEM1")
    ap.add_argument("--split-file", type=Path, default=None, help="default: <data-dir>/subject_split.json")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--px-nm", type=float, nargs="+", default=[2.36, 4.93, 7, 10, 16, 21.3, 28.3])
    ap.add_argument("--max-images", type=int, default=None, help="first N images only (smoke test)")
    ap.add_argument("--min-axon-diam-um", type=float, default=0.0,
                    help="drop axons with a smaller equivalent diameter (0 keeps all)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    logging.disable(logging.WARNING)

    split = json.loads((args.split_file or args.data_dir / "subject_split.json").read_text())
    images = sorted(p.name for p in (args.results_dir / args.models[0]).iterdir()
                    if p.is_dir() and p.name.split("_")[0] in split["test_subjects"])
    if args.max_images:
        images = images[: args.max_images]
    print(f"{len(images)} images, models {args.models}, sizes {args.px_nm} nm", flush=True)
    jobs = [(i, args.results_dir, args.data_dir, args.models, args.px_nm, args.min_axon_diam_um) for i in images]
    with Pool(args.workers) as pool:
        rows = [r for rs in pool.imap_unordered(one_image, jobs) for r in rs]
    df = pd.DataFrame(rows).sort_values(["image", "source", "px_nm"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"-> {args.out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
