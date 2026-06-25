#!/usr/bin/env python3
"""
Compute ADS morphometrics on a 10% sample of TEM1 and TEM2 prediction masks,
across all 16 pixel sizes, for witness and multires models.

Outputs:
  morphometrics_tem1.csv
  morphometrics_tem2.csv
  morphometrics_tem1.png
  morphometrics_tem2.png
"""

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

from AxonDeepSeg.morphometrics.compute_morphometrics import get_axon_morphometrics

RESULTS_TEM1 = Path("/Users/yolaatar/Developer/ADS/resinv/results_nnunet")
RESULTS_TEM2 = Path("/Users/yolaatar/Developer/ADS/resinv/results_nnunet_tem2")
OUT_DIR      = Path("/Users/yolaatar/Developer/ADS/resinv")

MODELS = ["witness", "multires"]

# 10% sample — 3 TEM1, 4 TEM2
SAMPLE_TEM1 = [
    "sub-nyuMouse07_sample-0001_TEM",
    "sub-nyuMouse11_sample-0004_TEM",
    "sub-nyuMouse25_sample-0002_TEM",
]
SAMPLE_TEM2 = [
    "sub-370_sample-0001_TEM",
    "sub-372_sample-0004_TEM",
    "sub-374_sample-0002_TEM",
    "sub-375_sample-0006_TEM",
]

MODEL_COLORS = {"witness": "#d62728", "multires": "#1f77b4"}


def parse_px(filename: str) -> float | None:
    m = re.search(r"_px([\d.e+-]+)um_", filename)
    return float(m.group(1)) if m else None


def run_morphometrics(pred_dir: Path, px: float) -> dict | None:
    tag = f"_px{px:.7g}um_"
    axon_files = list(pred_dir.glob(f"*{tag}seg-axon.png"))
    myelin_files = list(pred_dir.glob(f"*{tag}seg-myelin.png"))
    if not axon_files or not myelin_files:
        return None

    im_axon  = np.array(Image.open(axon_files[0]).convert("L")) > 127
    im_myelin = np.array(Image.open(myelin_files[0]).convert("L")) > 127

    if im_axon.sum() == 0:
        return None

    try:
        df = get_axon_morphometrics(
            im_axon,
            im_myelin=im_myelin,
            pixel_size=px,
            axon_shape="circle",
            return_border_info=True,
        )
    except Exception as e:
        print(f"    ERROR: {e}")
        return None

    # Filter border-touching and gratio == 1
    if "image_border_touching" in df.columns:
        df = df[~df["image_border_touching"].fillna(True).astype(bool)]
    if "gratio" in df.columns:
        df = df[df["gratio"] < 1.0]
    if "axon_diam" in df.columns:
        df = df[df["axon_diam"] >= 0.3]

    if len(df) == 0:
        return None

    return {
        "n_axons":      len(df),
        "mean_diam":    df["axon_diam"].mean() if "axon_diam" in df.columns else np.nan,
        "std_diam":     df["axon_diam"].std()  if "axon_diam" in df.columns else np.nan,
        "mean_gratio":  df["gratio"].mean()    if "gratio" in df.columns else np.nan,
        "std_gratio":   df["gratio"].std()     if "gratio" in df.columns else np.nan,
    }


def process_dataset(results_dir: Path, sample_images: list, dataset_name: str) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for img in sample_images:
            pred_dir = results_dir / model / img / "predictions"
            if not pred_dir.exists():
                print(f"  Missing: {model}/{img}")
                continue
            print(f"  {model} / {img}")

            all_px = sorted({
                parse_px(p.name)
                for p in pred_dir.glob("*_seg-axon.png")
                if parse_px(p.name) is not None
            })

            for px in all_px:
                metrics = run_morphometrics(pred_dir, px)
                if metrics is None:
                    continue
                rows.append({
                    "dataset":    dataset_name,
                    "model":      model,
                    "image":      img,
                    "pixel_size": px,
                    **metrics,
                })
                print(f"    {px:.5g} μm/px — n={metrics['n_axons']}  diam={metrics['mean_diam']:.3f}  g={metrics['mean_gratio']:.3f}")

    return pd.DataFrame(rows)


def plot_morphometrics(df: pd.DataFrame, dataset_name: str, out_path: Path, native_px: float):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Morphometrics across pixel sizes — {dataset_name} (10% sample)", fontsize=12, fontweight="bold")

    for ax, metric, ylabel, std_col in zip(
        axes,
        ["mean_diam", "mean_gratio"],
        ["Mean axon diameter [μm]", "Mean g-ratio"],
        ["std_diam",  "std_gratio"],
    ):
        for model in MODELS:
            sub = df[df["model"] == model].groupby("pixel_size").agg(
                mean=(metric, "mean"),
                std=(metric, "std"),
            ).reset_index()
            color = MODEL_COLORS[model]
            ax.plot(sub["pixel_size"], sub["mean"], color=color, linewidth=2,
                    marker="o", markersize=4, label=model)
            ax.fill_between(sub["pixel_size"],
                            sub["mean"] - sub["std"],
                            sub["mean"] + sub["std"],
                            color=color, alpha=0.15)

        ax.axvline(native_px, color="gray", linestyle=":", linewidth=1, alpha=0.8, label="native px")
        ax.set_xscale("log")
        ax.set_xticks([0.0018625, 0.003, 0.00493, 0.007, 0.01, 0.016])
        ax.set_xticklabels(["1.9 nm", "3 nm", "4.9 nm", "7 nm", "10 nm", "16 nm"],
                           fontsize=8, rotation=45, ha="right")
        ax.set_xlim(0.0018625 * 0.9, 0.016 * 1.1)
        ax.set_xlabel("Pixel size", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, which="major", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== TEM1 ===")
    df_tem1 = process_dataset(RESULTS_TEM1, SAMPLE_TEM1, "TEM1")
    df_tem1.to_csv(OUT_DIR / "morphometrics_tem1.csv", index=False)
    plot_morphometrics(df_tem1, "TEM1", OUT_DIR / "morphometrics_tem1.png", native_px=0.00236)

    print("\n=== TEM2 ===")
    df_tem2 = process_dataset(RESULTS_TEM2, SAMPLE_TEM2, "TEM2")
    df_tem2.to_csv(OUT_DIR / "morphometrics_tem2.csv", index=False)
    plot_morphometrics(df_tem2, "TEM2", OUT_DIR / "morphometrics_tem2.png", native_px=0.00493)

    print("\nDone.")
