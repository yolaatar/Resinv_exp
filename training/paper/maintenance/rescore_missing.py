"""Rescore only the images whose axon or myelin Dice is empty in results.csv, with the fixed
recompute_metrics.py (a class with GT is always scored), and replace those rows in place.

Run once on 2026-11-05 on tassan: 12 images (nyuMouse25 0001 / 0006) of multires and canonical
resampling, whose axon class the old > 50 % coverage filter had skipped. Backups of the previous
files: results_old_20261105.csv next to each results.csv. Kept for the record.
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path.home() / "resinv_exp/scripts_paper"))
import recompute_metrics as rm  # noqa: E402

R = Path.home() / "resinv_exp/results_paper/tem1"
DATA = Path.home() / "resinv_exp/data/TEM1"
MODELS = [f"{m}_f{k}" for m in ("control", "multires", "control_canonical") for k in range(5)] + ["multires2_f0", "multires_to7_f0"]


def work(args):
    model, img = args
    rm.GT_LABELS = {"axon", "myelin"}
    return model, img, rm.recompute_image(R / model / img, model, DATA, True)


if __name__ == "__main__":
    tables, jobs = {}, []
    for m in MODELS:
        d = pd.read_csv(R / m / "results.csv")
        bad = sorted(d.loc[d.dice_axon_native.isna() | d.dice_myelin_native.isna(), "image"].unique())
        tables[m] = d
        jobs += [(m, i) for i in bad]
        print(m, "images to rescore:", bad, flush=True)
    with Pool(4) as pool:
        for m, img, new in pool.imap_unordered(work, jobs):
            d = tables[m]
            tables[m] = pd.concat([d[d.image != img], new], ignore_index=True)
            print(f"done {m} {img}: {len(new)} rows, empty axon {new.dice_axon_native.isna().sum()}", flush=True)
    for m, d in tables.items():
        d = d.sort_values(["image", "pixel_size_um"]).reset_index(drop=True)
        assert d.dice_axon_native.notna().all() and d.dice_myelin_native.notna().all(), m
        d.to_csv(R / m / "results.csv", index=False)
    print("ALL DONE", flush=True)
