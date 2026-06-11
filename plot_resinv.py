#!/usr/bin/env python3
"""
Plot resolution invariance results.
Can plot a single model or compare multiple models.

Usage:
    # After running evaluate_resinv.py for one or more models:
    python plot_resinv.py --results-dir ./results

    # Only specific models:
    python plot_resinv.py --results-dir ./results --models cambridge_unmyelinated generalist
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


TRAINING_PX = 0.00493


def _detect_labels(df: pd.DataFrame) -> tuple[str, str | None]:
    """Infer primary and secondary label names from CSV column names."""
    native_cols = [c for c in df.columns if c.endswith("_native") and c.startswith("dice_")]
    labels = [c.removeprefix("dice_").removesuffix("_native") for c in native_cols]
    primary = labels[0] if labels else "axon"
    secondary = labels[1] if len(labels) > 1 and df[f"dice_{labels[1]}_native"].notna().any() else None
    return primary, secondary


def plot_single_model(df: pd.DataFrame, model_name: str, out_path: Path):
    """Per-model plot: native and resized Dice across resolutions, per image."""
    primary, secondary = _detect_labels(df)
    has_secondary = secondary is not None
    n_rows = 2 if has_secondary else 1
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(f"Resolution Invariance — {model_name}", fontsize=14, y=1.01)

    panel_configs = [
        (f"dice_{primary}_native", f"dice_{primary}_interp_baseline",
         f"{primary} — Native space (vs reference at {TRAINING_PX} μm/px)", 0),
        (f"dice_{primary}_resized", None,
         f"{primary} — Resized space (vs reference downsampled to same res)", 1),
    ]
    if has_secondary:
        panel_configs += [
            (f"dice_{secondary}_native", f"dice_{secondary}_interp_baseline",
             f"{secondary} — Native space", 0),
            (f"dice_{secondary}_resized", None,
             f"{secondary} — Resized space", 1),
        ]

    for i, (metric, interp_metric, title, col) in enumerate(panel_configs):
        row = i // 2 if has_secondary else 0
        ax = axes[row, col]

        plot_df = df[df[metric].notna()].copy()
        sns.lineplot(data=plot_df, x="pixel_size_um", y=metric,
                     hue="image", marker="o", ax=ax)

        if interp_metric and df[interp_metric].notna().any():
            interp_df = df[df[interp_metric].notna()].copy()
            avg_interp = interp_df.groupby("pixel_size_um")[interp_metric].mean().reset_index()
            ax.plot(avg_interp["pixel_size_um"], avg_interp[interp_metric],
                    "k--", alpha=0.5, marker="x", label="Interp. baseline (avg)")
            ax.legend(fontsize=8)

        ax.axvline(x=TRAINING_PX, color="gray", linestyle=":", alpha=0.7)
        ax.text(TRAINING_PX * 1.05, 0.52, "train res", fontsize=8,
                color="gray", va="bottom")
        ax.set_xscale("log")
        ax.set_xlabel("Pixel size (μm/px)")
        ax.set_ylabel("Dice score")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0.5, 1.05)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_comparison(model_dfs: list[tuple[str, pd.DataFrame]], out_path: Path):
    """Compare multiple models: average Dice per model across images."""
    combined = pd.concat(
        [df.assign(model=name) for name, df in model_dfs],
        ignore_index=True,
    )
    # Collect all dice_*_native / dice_*_resized columns present across all models
    dice_cols = [c for c in combined.columns
                 if c.startswith("dice_") and (c.endswith("_native") or c.endswith("_resized"))]
    avg = combined.groupby(["model", "pixel_size_um"])[dice_cols].mean().reset_index()

    configs = []
    seen_labels = []
    for col in dice_cols:
        label = col.removeprefix("dice_").removesuffix("_native").removesuffix("_resized")
        if label not in seen_labels:
            seen_labels.append(label)
    for label in seen_labels:
        configs.append((f"dice_{label}_native", f"{label} — Native space"))
        configs.append((f"dice_{label}_resized", f"{label} — Resized space"))

    # Only keep configs where column actually exists in avg
    configs = [(col, title) for col, title in configs if col in avg.columns]

    n_cols = len(configs)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]
    fig.suptitle("Resolution Invariance — Model Comparison", fontsize=14)

    for ax, (metric, title) in zip(axes, configs):
        plot_df = avg[avg[metric].notna()].copy()
        sns.lineplot(data=plot_df, x="pixel_size_um", y=metric,
                     hue="model", marker="o", ax=ax)
        ax.axvline(x=TRAINING_PX, color="gray", linestyle=":", alpha=0.7)
        ax.set_xscale("log")
        ax.set_xlabel("Pixel size (μm/px)")
        ax.set_ylabel("Dice score (avg across images)")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0.5, 1.05)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot resolution invariance results")
    parser.add_argument("--results-dir", type=Path, default=Path("./results"),
                        help="Root results directory (default: ./results)")
    parser.add_argument("--models", type=str, nargs="*",
                        help="Model names to include (default: all found)")
    args = parser.parse_args()

    model_dfs = []
    for csv_path in sorted(args.results_dir.glob("*/results.csv")):
        model_name = csv_path.parent.name
        if args.models and model_name not in args.models:
            continue
        df = pd.read_csv(csv_path)
        model_dfs.append((model_name, df))
        plot_single_model(df, model_name, csv_path.parent / "results.png")

    if not model_dfs:
        print(f"No results.csv files found under {args.results_dir}")
        return

    if len(model_dfs) > 1:
        plot_comparison(model_dfs, args.results_dir / "comparison.png")


if __name__ == "__main__":
    main()
