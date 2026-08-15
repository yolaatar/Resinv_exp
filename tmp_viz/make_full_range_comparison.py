import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

TRAINING_PX = 0.00493
XTICKS_UM = [0.001, 0.002, 0.00493, 0.008, 0.016, 0.03, 0.05]
XTICK_LABELS = ["1 nm", "2 nm", "4.9 nm (train)", "8 nm", "16 nm", "30 nm", "50 nm"]

models = {
    "witness": pd.read_csv("results_nnunet/witness/results.csv"),
    "multires": pd.read_csv("results_nnunet/multires/results.csv"),
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
colors = {"witness": "tab:blue", "multires": "tab:orange"}

for ax, label in zip(axes, ["axon", "myelin"]):
    metric = f"dice_{label}_native"
    for name, df in models.items():
        avg = df.dropna(subset=[metric]).groupby("pixel_size_um")[metric].mean().reset_index()
        ax.plot(avg["pixel_size_um"], avg[metric], color=colors[name], linewidth=2,
                marker="o", markersize=4, label=name)
    ax.axvline(x=TRAINING_PX, color="gray", linestyle=":", linewidth=1, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xticks(XTICKS_UM)
    ax.set_xticklabels(XTICK_LABELS, fontsize=8, rotation=45, ha="right")
    ax.set_xlim(XTICKS_UM[0]*0.9, XTICKS_UM[-1]*1.1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Pixel size", fontsize=9)
    ax.set_ylabel("Dice (mean across images)", fontsize=9)
    ax.set_title(f"{label} (native referential)", fontsize=11)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(fontsize=9)

fig.suptitle("Resolution Invariance: witness vs multires — full range (1-50 nm/px)", fontsize=13)
plt.tight_layout()
plt.savefig("results_nnunet/comparison_full_range.png", dpi=150, bbox_inches="tight")
print("Saved results_nnunet/comparison_full_range.png")
