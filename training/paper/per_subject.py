#!/usr/bin/env python3
"""
Per-subject results on the TEM1 test set: for each test mouse, model, fold and pixel size, the mean
over that mouse's images of axon, myelin and mean (axon + myelin) / 2 Dice, from results.csv.

With four test subjects, subject-level spread is what tells whether a result holds for every animal
or is carried by one. Prints, per model, the five-fold mean Dice of each subject at a few pixel sizes
and each subject's robust band (mean Dice >= 0.89, the paper's definition).

Usage (tassan, CPU, seconds):
  python per_subject.py --out ~/resinv_exp/summaries/per_subject_tem1.csv
"""

import argparse
from pathlib import Path

import pandas as pd

HOME = Path.home()
MODELS = {"control": "control", "multires": "multires", "canonical": "control_canonical"}
THRESHOLD = 0.89
SHOW = [1.0, 2.36, 4.93, 7.0, 10.0, 16.0, 21.3, 28.3, 50.0]


def bands(s: pd.Series) -> str:
    """Contiguous runs of the pixel-size grid where the value is >= THRESHOLD."""
    runs, start, prev = [], None, None
    for nm, v in s.sort_index().items():
        if v >= THRESHOLD and start is None:
            start = nm
        if v < THRESHOLD and start is not None:
            runs.append((start, prev)); start = None
        prev = nm
    if start is not None:
        runs.append((start, prev))
    return ", ".join(f"{a:g}-{b:g}" for a, b in runs) or "none"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results-dir", type=Path, default=HOME / "resinv_exp/results_paper/tem1")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    parts = []
    for key, folder in MODELS.items():
        for fold in range(5):
            f = args.results_dir / f"{folder}_f{fold}" / "results.csv"
            d = pd.read_csv(f)
            assert d.dice_axon_native.notna().all() and d.dice_myelin_native.notna().all(), f
            d["subject"] = d.image.str.split("_").str[0]
            d["nm"] = (d.pixel_size_um * 1000).round(2)
            d["dice_mean"] = (d.dice_axon_native + d.dice_myelin_native) / 2
            g = d.groupby(["subject", "nm"]).agg(n_images=("image", "nunique"), axon=("dice_axon_native", "mean"),
                                                 myelin=("dice_myelin_native", "mean"), mean=("dice_mean", "mean"))
            g = g.reset_index()
            g["model"], g["fold"] = key, fold
            parts.append(g)
    a = pd.concat(parts, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    a.round(4).to_csv(args.out, index=False)

    five = a.groupby(["model", "subject", "nm"])["mean"].agg(["mean", "std"]).reset_index()
    print("Images per subject:", a.groupby("subject").n_images.max().to_dict())
    for key in MODELS:
        t = five[five.model == key].pivot(index="subject", columns="nm", values="mean")
        print(f"\n{key}: mean Dice per subject (five-fold mean)")
        print(t[[c for c in SHOW if c in t.columns]].round(3).to_string())
        for subj, row in t.iterrows():
            print(f"  {subj}: robust band {bands(row)}")
        spread = t.max() - t.min()
        print(f"  between-subject range at 2.36 / 7 / 16 / 28.3 nm: "
              + " / ".join(f"{spread.get(nm, float('nan')):.3f}" for nm in (2.36, 7.0, 16.0, 28.3)))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
