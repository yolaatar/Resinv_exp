"""
Visualize the control model's predictions across the pixel-size range where its
Dice score is high (~2-8.3 nm/px), to sanity-check that the "it works fine near
training resolution" result is a real correct segmentation and not a metrics
artifact.

Crops a fixed physical patch (in um) from one test image, centered, at each
pixel size in the "good" range, and overlays axon/myelin/uaxon predictions.

Usage:
    python viz_control_working_range.py
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = None

BASE = r"C:\Users\Youssef\Documents\Cours\ADS\ResInv_exp\results_armand_uaxon\armand_control"
IMG_NAME = "sub-366A_sample-0001_acq-roi_TEM"
PRED_DIR = f"{BASE}/{IMG_NAME}/predictions"
OUT = r"C:\Users\Youssef\Documents\Cours\ADS\ResInv_exp\results_armand_uaxon"

# "good" range: Dice_axon_native >= ~0.96 for armand_control on this dataset
GOOD_PX_UM = [0.002, 0.00236, 0.0027058, 0.0032614, 0.003931, 0.004738, 0.00493, 0.0057108, 0.0068833, 0.0082966]

PATCH_UM = 6.0  # physical size of the cropped patch, same for every pixel size

COLORS = {
    "axon": (255, 0, 0),      # red
    "myelin": (0, 255, 255),  # cyan
    "uaxon": (255, 255, 0),   # yellow
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


n = len(GOOD_PX_UM)
fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.6))

for i, px in enumerate(GOOD_PX_UM):
    tag = px_tag(px)
    raw = load_gray(f"{PRED_DIR}/{IMG_NAME}_{tag}.png")
    masks = {}
    for label in ["axon", "myelin", "uaxon"]:
        p = f"{PRED_DIR}/{IMG_NAME}_{tag}_seg-{label}.png"
        masks[label] = load_gray(p)

    patch_px = round(PATCH_UM / px)
    raw_c = center_crop(raw, patch_px)
    masks_c = {k: center_crop(v, patch_px) for k, v in masks.items()}
    ov = overlay(raw_c, masks_c)

    axes[0, i].imshow(raw_c, cmap="gray", vmin=0, vmax=255)
    axes[0, i].set_title(f"{px*1000:g} nm/px\n({raw_c.shape[1]}x{raw_c.shape[0]}px)", fontsize=9)
    axes[0, i].axis("off")

    axes[1, i].imshow(ov)
    axes[1, i].axis("off")

axes[0, 0].set_ylabel("raw", fontsize=10)
axes[1, 0].set_ylabel("prediction overlay", fontsize=10)

legend_patches = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=np.array(c)/255,
                              markersize=12, label=l) for l, c in COLORS.items()]
fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=10, frameon=False)

fig.suptitle(f"armand_control — {IMG_NAME}, {PATCH_UM:g}x{PATCH_UM:g} um patch\n"
             f"pixel sizes where Dice is high (~2-8.3 nm/px)", fontsize=12)
plt.tight_layout(rect=[0, 0.05, 1, 0.94])
plt.savefig(f"{OUT}/control_working_range_predictions.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT}/control_working_range_predictions.png")
