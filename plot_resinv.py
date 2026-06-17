#!/usr/bin/env python3
"""
Plot resolution invariance results.

Usage:
    python plot_resinv.py --results-dir ./results
    python plot_resinv.py --results-dir ./results --models unmyelinated_tem
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


TRAINING_PX = 0.00493
ORIGINAL_PX = 0.0018625


def _detect_labels(df: pd.DataFrame) -> list[str]:
    """Return all labels that have at least some non-trivial Dice data."""
    native_cols = [c for c in df.columns if c.startswith("dice_") and c.endswith("_native")]
    labels = [c.removeprefix("dice_").removesuffix("_native") for c in native_cols]
    # Keep only labels where not all values are 0 or 1 (i.e., actually varying)
    active = []
    for l in labels:
        col = df[f"dice_{l}_native"].dropna()
        if col.nunique() > 1:
            active.append(l)
    return active if active else labels


# Meaningful tick positions in μm/px and their nm labels
_XTICKS_UM = [0.0018625, 0.003, 0.00493, 0.007, 0.01, 0.016]
_XTICK_LABELS = ["1.9 nm", "3 nm", "4.9 nm (train)", "7 nm", "10 nm", "16 nm"]


def _set_x_ticks(ax):
    ax.set_xticks(_XTICKS_UM)
    ax.set_xticklabels(_XTICK_LABELS, fontsize=8, rotation=45, ha="right")
    ax.set_xlim(_XTICKS_UM[0] * 0.9, _XTICKS_UM[-1] * 1.1)


def _add_training_line(ax):
    ax.axvline(x=TRAINING_PX, color="gray", linestyle=":", linewidth=1, alpha=0.8)


def plot_single_model(
    df: pd.DataFrame,
    model_name: str,
    out_path: Path,
    sample_n: int | None = None,
    exclude_labels: list[str] | None = None,
    show_resized: bool = False,
):
    labels = _detect_labels(df)
    if exclude_labels:
        labels = [l for l in labels if l not in exclude_labels]
    df = df.copy()

    # Build panel list: for each label, native first; if show_resized, add resized alongside
    if show_resized:
        panel_configs = []
        for label in labels:
            panel_configs.append((label, "native",   f"{label}\n(native)"))
            panel_configs.append((label, "resized",  f"{label}\n(resized)"))
    else:
        panel_configs = [(label, "native", label) for label in labels]

    n_cols = len(panel_configs)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 5), sharey=True)
    if n_cols == 1:
        axes = [axes]

    fig.suptitle(f"Resolution Invariance: {model_name}", fontsize=13)

    all_images = df["image"].unique()
    # Subsample for display; mean is always over all images
    if sample_n is not None and sample_n < len(all_images):
        idx = np.round(np.linspace(0, len(all_images) - 1, sample_n)).astype(int)
        display_images = all_images[idx]
    else:
        display_images = all_images
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(display_images)))
    img_color = {img: colors[i] for i, img in enumerate(display_images)}

    for i, (label, space, title) in enumerate(panel_configs):
        ax = axes[i]
        metric = f"dice_{label}_{space}"
        interp = f"dice_{label}_interp_baseline"

        if metric not in df.columns:
            ax.set_visible(False)
            continue

        x_col = "pixel_size_um"

        # Per-image lines (thin, sampled subset only)
        for img in display_images:
            sub = df[df["image"] == img].dropna(subset=[metric]).sort_values(x_col)
            ax.plot(sub[x_col], sub[metric], color=img_color[img],
                    alpha=0.5, linewidth=0.9, marker="o", markersize=2)

        # Mean across ALL images (bold)
        avg = df.dropna(subset=[metric]).groupby(x_col)[metric].mean().reset_index()
        ax.plot(avg[x_col], avg[metric], color="black",
                linewidth=2, marker="o", markersize=4, label=f"mean (n={len(all_images)})", zorder=5)

        # Highlight peak of the mean
        peak_idx = avg[metric].idxmax()
        peak_x, peak_y = avg.loc[peak_idx, x_col], avg.loc[peak_idx, metric]
        ax.plot(peak_x, peak_y, marker="*", color="crimson", markersize=12, zorder=6)
        ax.annotate(f"{peak_y:.2f}", xy=(peak_x, peak_y),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color="crimson", fontweight="bold")

        # Interpolation baseline (dashed, mean only)
        if interp in df.columns and df[interp].notna().any():
            avg_interp = df.dropna(subset=[interp]).groupby(x_col)[interp].mean().reset_index()
            ax.plot(avg_interp[x_col], avg_interp[interp], color="black",
                    linewidth=1.5, linestyle="--", alpha=0.45, label="interp. baseline", zorder=4)

        _add_training_line(ax)
        ax.set_xscale("log")
        _set_x_ticks(ax)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Pixel size", fontsize=9)
        ax.set_ylabel("Dice", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.grid(True, which="major", alpha=0.3)
        ax.grid(True, which="minor", alpha=0.1)
        ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_comparison(model_dfs: list[tuple[str, pd.DataFrame]], out_path: Path):
    """Compare models: mean Dice per model, native referential only."""
    # Collect all label/space combos present
    all_labels = set()
    for _, df in model_dfs:
        for l in _detect_labels(df):
            all_labels.add(l)

    configs = [(l, "native") for l in sorted(all_labels)]
    n_cols = len(configs)

    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), sharey=True)
    if n_cols == 1:
        axes = [axes]
    fig.suptitle("Resolution Invariance: Model Comparison (native referential)", fontsize=13)

    colors = plt.cm.Set1(np.linspace(0, 0.7, len(model_dfs)))

    for ax, (label, space) in zip(axes, configs):
        metric = f"dice_{label}_{space}"
        interp = f"dice_{label}_interp_baseline"

        for (model_name, df), color in zip(model_dfs, colors):
            if metric not in df.columns:
                continue
            x_col = "pixel_size_um"
            avg = df.dropna(subset=[metric]).groupby(x_col)[metric].mean().reset_index()
            ax.plot(avg[x_col], avg[metric], color=color, linewidth=2,
                    marker="o", markersize=4, label=model_name)

            if interp in df.columns and df[interp].notna().any():
                avg_i = df.dropna(subset=[interp]).groupby(x_col)[interp].mean().reset_index()
                ax.plot(avg_i[x_col], avg_i[interp], color=color,
                        linewidth=1.5, linestyle="--", alpha=0.4)

        _add_training_line(ax)
        ax.set_xscale("log")
        _set_x_ticks(ax)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Pixel size", fontsize=9)
        ax.set_ylabel("Dice (mean across images)", fontsize=9)
        ax.set_title(f"{label} (native referential)", fontsize=10)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.grid(True, which="major", alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("./results"))
    parser.add_argument("--models", type=str, nargs="*")
    parser.add_argument("--sample-n", type=int, default=None,
                        help="Show only N evenly-sampled per-image lines (mean is always over all images)")
    parser.add_argument("--exclude-labels", type=str, nargs="*", default=None,
                        help="Labels to hide from the plot, e.g. --exclude-labels uaxon")
    parser.add_argument("--show-resized", action="store_true",
                        help="Show native and resized panels side by side for each label")
    parser.add_argument("--out-suffix", type=str, default="",
                        help="Suffix appended to output filename, e.g. '_all' -> results_all.png")
    args = parser.parse_args()

    model_dfs = []
    for csv_path in sorted(args.results_dir.glob("*/results.csv")):
        model_name = csv_path.parent.name
        if args.models and model_name not in args.models:
            continue
        df = pd.read_csv(csv_path)
        model_dfs.append((model_name, df))
        out_name = f"results{args.out_suffix}.png"
        plot_single_model(
            df, model_name, csv_path.parent / out_name,
            sample_n=args.sample_n,
            exclude_labels=args.exclude_labels,
            show_resized=args.show_resized,
        )

    if not model_dfs:
        print(f"No results.csv found under {args.results_dir}")
        return

    if len(model_dfs) > 1:
        plot_comparison(model_dfs, args.results_dir / "comparison.png")


if __name__ == "__main__":
    main()
