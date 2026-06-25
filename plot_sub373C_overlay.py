#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

BASE       = Path("/Users/yolaatar/Developer/ADS/data/TEM2")
PRED_BASE  = Path("/Users/yolaatar/Developer/ADS/resinv/results_nnunet_tem2/multires")
OUT        = Path("/Users/yolaatar/Developer/ADS/resinv/sub373C_overlay.png")

CHUNKS = [
    "sub-373C_sample-0001_acq-roi_chunk-01_TEM",
    "sub-373C_sample-0001_acq-roi_chunk-02_TEM",
]

def load_gray(path):
    return np.array(Image.open(path).convert("L"))

def overlay(ax, img, masks_colors, title):
    ax.imshow(img, cmap="gray", interpolation="nearest")
    for mask, color, alpha in masks_colors:
        rgba = np.zeros((*img.shape, 4), dtype=np.float32)
        rgba[mask > 127, :3] = color
        rgba[mask > 127, 3]  = alpha
        ax.imshow(rgba, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("sub-373C acq-roi — multires predictions vs GT (native px 4.93 nm)", fontsize=12, fontweight="bold")

for row, chunk in enumerate(CHUNKS):
    img_path = BASE / "sub-373C" / "micr" / f"{chunk}.png"
    gt_path  = BASE / "derivatives" / "labels" / "sub-373C" / "micr" / f"{chunk}_seg-axonmyelin-manual.png"
    pred_axon   = PRED_BASE / chunk / "predictions" / f"{chunk}_px0.00493um_seg-axon.png"
    pred_myelin = PRED_BASE / chunk / "predictions" / f"{chunk}_px0.00493um_seg-myelin.png"

    img    = load_gray(img_path)
    gt_raw  = np.array(Image.open(gt_path).convert("L"))
    gt_axon  = (gt_raw >= 200).astype(np.uint8) * 255
    gt_myelin = ((gt_raw >= 100) & (gt_raw < 200)).astype(np.uint8) * 255
    p_axon = load_gray(pred_axon)
    p_mye  = load_gray(pred_myelin)

    chunk_label = chunk.split("chunk-")[1].split("_")[0]

    # Raw image
    axes[row, 0].imshow(img, cmap="gray", interpolation="nearest")
    axes[row, 0].set_title(f"Chunk-0{chunk_label} — raw image", fontsize=9)
    axes[row, 0].axis("off")

    # GT overlay (axon=red, myelin=blue)
    overlay(axes[row, 1], img,
            [(gt_myelin, [0.2, 0.4, 1.0], 0.45),
             (gt_axon,   [1.0, 0.15, 0.15], 0.55)],
            f"Chunk-0{chunk_label} — GT (axon=red, myelin=blue)")

    # Prediction overlay (axon=red, myelin=blue)
    overlay(axes[row, 2], img,
            [(p_mye,  [0.2, 0.4, 1.0], 0.45),
             (p_axon, [1.0, 0.15, 0.15], 0.55)],
            f"Chunk-0{chunk_label} — multires pred (axon=red, myelin=blue)")

plt.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT}")
