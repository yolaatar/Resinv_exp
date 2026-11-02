"""Compare armand_uaxon (multires) vs armand_control (v3.0.0 baseline) across pixel sizes.

Usage:
    python plot_comparison.py
"""
import pandas as pd
import matplotlib.pyplot as plt

BASE = r"C:\Users\Youssef\Documents\Cours\ADS\ResInv_exp\results_armand_uaxon"
OUT = BASE

MODELS = {
    "armand_uaxon": {"label": "multires", "color": "#0072B2"},
    "armand_control": {"label": "control (v3.0.0)", "color": "#D55E00"},
}

GRID = "#dddddd"
TRAINING_PX_NM = 4.93


def style_ax(ax, ylabel, title, ticks=None):
    ax.set_xlabel("pixel size (nm/px)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_xscale("log")
    if ticks:
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.legend(fontsize=8, frameon=False)


def load(model_key):
    d = {}
    d["results"] = pd.read_csv(f"{BASE}/{model_key}/results.csv")
    d["morph"] = pd.read_csv(f"{BASE}/{model_key}/morphometrics_summary.csv")
    d["counts"] = pd.read_csv(f"{BASE}/{model_key}/axon_counts_all.csv")
    for k in d:
        d[k]["pixel_size_nm"] = d[k]["pixel_size_um"] * 1000
    return d


data = {m: load(m) for m in MODELS}

# ---------------------------------------------------------------------------
# Figure 1: Dice (axon, myelin, uaxon) — native metric, mean +/- std across images
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, label in zip(axes, ["axon", "myelin", "uaxon"]):
    for m, cfg in MODELS.items():
        r = data[m]["results"]
        g = r.groupby("pixel_size_nm")[f"dice_{label}_native"].agg(["mean", "std"]).reset_index()
        ax.plot(g["pixel_size_nm"], g["mean"], "o-", color=cfg["color"], label=cfg["label"], markersize=3)
        ax.fill_between(g["pixel_size_nm"], g["mean"] - g["std"], g["mean"] + g["std"],
                         color=cfg["color"], alpha=0.15)
    ax.axvline(TRAINING_PX_NM, color=GRID, linewidth=1, zorder=0, linestyle="--")
    style_ax(ax, f"Dice ({label})", f"{label} Dice vs pixel size", ticks=[1, 2, 4.93, 8, 16, 30, 50])
    ax.set_ylim(0, 1.0)
fig.suptitle("Multires vs control model — Dice score (native)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT}/dice_comparison.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT}/dice_comparison.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: G-ratio — zoomed (<16nm) and full range
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

all_pxs = sorted(data["armand_uaxon"]["morph"]["pixel_size_nm"].unique())
zoom_pxs = sorted(p for p in all_pxs if p < 16)
zoom_max = max(zoom_pxs)
zoom_ticks = sorted(set([1, 2, 4.93, 8, round(zoom_max, 3)]))

ax = axes[0]
for m, cfg in MODELS.items():
    g = data[m]["morph"].groupby("pixel_size_nm")["mean_gratio"].mean().reset_index()
    mask = g["pixel_size_nm"].isin(zoom_pxs)
    ax.plot(g.loc[mask, "pixel_size_nm"], g.loc[mask, "mean_gratio"], "o-",
            color=cfg["color"], label=cfg["label"], markersize=3)
ax.axvline(TRAINING_PX_NM, color=GRID, linewidth=1, zorder=0, linestyle="--")
style_ax(ax, "mean g-ratio", f"G-ratio, zoomed (\u2264{zoom_max:.3g}nm/px)", ticks=zoom_ticks)
ax.set_xlim(min(zoom_pxs) * 0.9, zoom_max * 1.05)
ax.set_ylim(0.5, 1.0)

ax = axes[1]
for m, cfg in MODELS.items():
    g = data[m]["morph"].groupby("pixel_size_nm")["mean_gratio"].mean().reset_index()
    ax.plot(g["pixel_size_nm"], g["mean_gratio"], "o-", color=cfg["color"], label=cfg["label"], markersize=3)
ax.axhline(1.0, color=GRID, linewidth=1, zorder=0)
style_ax(ax, "mean g-ratio", "G-ratio (1.0 = no myelin detected)", ticks=[1, 2, 4.93, 8, 16, 30, 50])

fig.suptitle("Multires vs control model — g-ratio", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT}/gratio_comparison.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT}/gratio_comparison.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3: axon / uaxon counts — full range and zoomed (drop 1nm point, 10-90 y-range)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

count_zoom_pxs = [p for p in all_pxs if 1 < p < 16]
count_zoom_min, count_zoom_max = min(count_zoom_pxs), max(count_zoom_pxs)

for row, count_col, title in [(0, "axon_count", "axon count"), (1, "uaxon_count", "uaxon count")]:
    ax = axes[row, 0]
    for m, cfg in MODELS.items():
        g = data[m]["counts"].groupby("pixel_size_nm")[count_col].mean().reset_index()
        ax.plot(g["pixel_size_nm"], g[count_col], "o-", color=cfg["color"], label=cfg["label"], markersize=3)
    ax.axvline(TRAINING_PX_NM, color=GRID, linewidth=1, zorder=0, linestyle="--")
    style_ax(ax, f"mean {title}", f"{title} (full range)", ticks=[1, 2, 4.93, 8, 16, 30, 50])
    ax.set_ylim(10, 90)

    ax = axes[row, 1]
    for m, cfg in MODELS.items():
        g = data[m]["counts"].groupby("pixel_size_nm")[count_col].mean().reset_index()
        mask = g["pixel_size_nm"].isin(count_zoom_pxs)
        ax.plot(g.loc[mask, "pixel_size_nm"], g.loc[mask, count_col], "o-",
                color=cfg["color"], label=cfg["label"], markersize=3)
    ax.axvline(TRAINING_PX_NM, color=GRID, linewidth=1, zorder=0, linestyle="--")
    zticks = sorted(set([1.5, 2, 4.93, 8, round(count_zoom_max, 3)]))
    style_ax(ax, f"mean {title}", f"{title}, zoomed ({count_zoom_min:.3g}\u2013{count_zoom_max:.3g}nm/px)", ticks=zticks)
    ax.set_xlim(count_zoom_min * 0.9, count_zoom_max * 1.05)
    ax.set_ylim(10, 90)

fig.suptitle("Multires vs control model — object counts", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT}/counts_comparison.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT}/counts_comparison.png")
plt.close(fig)

print("Done.")
