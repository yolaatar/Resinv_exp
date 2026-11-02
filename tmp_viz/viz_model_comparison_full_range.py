"""
Compare armand_uaxon (multires) vs armand_control (v3.0.0 baseline) predictions
on the same test image, across the full 1-50 nm/px range, so we can see exactly
where the two models diverge.

Crops a fixed physical patch (in um) from the same test image, centered, at
each pixel size, and overlays axon/myelin/uaxon predictions for both models.

Usage:
    python viz_model_comparison_full_range.py
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = None

BASE = r"C:\Users\Youssef\Documents\Cours\ADS\ResInv_exp\results_armand_uaxon"
IMG_NAME = "sub-366A_sample-0001_acq-roi_TEM"
OUT = BASE

MODELS = {
    "armand_uaxon": "multires",
    "armand_control": "control (v3.0.0)",
}

PX_SIZES = [
    0.001, 0.0015, 0.002,
    0.00236, 0.0027058, 0.0032614, 0.003931, 0.004738, 0.00493,
    0.0057108, 0.0068833, 0.0082966, 0.01, 0.0108148, 0.0116961,
    0.0126491, 0.0136798, 0.0147945, 0.016,
    0.020, 0.030, 0.050,
]

PATCH_UM = 6.0

COLORS = {
    "axon": (255, 0, 0),
    "myelin": (0, 255, 255),
    "uaxon": (255, 255, 0),
}


def px_tag(px):
    return f"px{px:.7g}um"


def load_gray(path):
    return np.array(Image.open(path).convert("L"))


def overlay(raw, masks):
    rgb = np.stack([raw, raw, raw], axis=-1).astype(np.float32)
    for label, mask in masks.items():
        color = np.array(COLORS[label], dtype=np.float32)
        alpha = 0.45
        m = mask > 0
        rgb[m] = (1 - alpha) * rgb[m] + alpha * color
    return rgb.astype(np.uint8)


def center_crop(arr, patch_px):
    h, w = arr.shape[:2]
    patch_px = min(patch_px, h, w)
    cy, cx = h // 2, w // 2
    y0, x0 = cy - patch_px // 2, cx - patch_px // 2
    return arr[y0:y0 + patch_px, x0:x0 + patch_px]


n = len(PX_SIZES)
fig, axes = plt.subplots(3, n, figsize=(2.0 * n, 6.5))

for i, px in enumerate(PX_SIZES):
    tag = px_tag(px)

    raw_path = f"{BASE}/armand_uaxon/{IMG_NAME}/predictions/{IMG_NAME}_{tag}.png"
    raw = load_gray(raw_path)
    patch_px = round(PATCH_UM / px)
    raw_c = center_crop(raw, patch_px)

    axes[0, i].imshow(raw_c, cmap="gray", vmin=0, vmax=255)
    axes[0, i].set_title(f"{px*1000:g} nm/px", fontsize=8)
    axes[0, i].axis("off")

    for row, model_key in [(1, "armand_uaxon"), (2, "armand_control")]:
        pred_dir = f"{BASE}/{model_key}/{IMG_NAME}/predictions"
        masks = {}
        for label in ["axon", "myelin", "uaxon"]:
            p = f"{pred_dir}/{IMG_NAME}_{tag}_seg-{label}.png"
            masks[label] = load_gray(p)
        masks_c = {k: center_crop(v, patch_px) for k, v in masks.items()}
        ov = overlay(raw_c, masks_c)
        axes[row, i].imshow(ov)
        axes[row, i].axis("off")

axes[0, 0].text(-0.35, 0.5, "raw", transform=axes[0, 0].transAxes,
                 fontsize=11, va="center", ha="right", rotation=90)
axes[1, 0].text(-0.35, 0.5, "multires", transform=axes[1, 0].transAxes,
                 fontsize=11, va="center", ha="right", rotation=90)
axes[2, 0].text(-0.35, 0.5, "control", transform=axes[2, 0].transAxes,
                 fontsize=11, va="center", ha="right", rotation=90)

legend_patches = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=np.array(c)/255,
                              markersize=12, label=l) for l, c in COLORS.items()]
fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=10, frameon=False)

fig.suptitle(f"multires vs control — {IMG_NAME}, {PATCH_UM:g}x{PATCH_UM:g} um patch, full pixel-size range", fontsize=13)
plt.tight_layout(rect=[0.02, 0.04, 1, 0.94])
plt.savefig(f"{OUT}/model_comparison_full_range.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT}/model_comparison_full_range.png")
