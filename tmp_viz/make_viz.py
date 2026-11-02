import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import resize

Image.MAX_IMAGE_PIXELS = None

ORIGINAL_PX = 0.0018625  # um/px, native TEM1 resolution (1.8625 nm)

img = np.array(Image.open("tmp_viz/sub-nyuMouse07_sample-0001_TEM.png").convert("L"))
h, w = img.shape

# crop a representative ~8um x 8um patch from the center (native px)
patch_um = 3.5
patch_px = int(round(patch_um / ORIGINAL_PX))
cy, cx = h // 2, w // 2
crop = img[cy - patch_px//2: cy + patch_px//2, cx - patch_px//2: cx + patch_px//2]

target_px_sizes_nm = [1.86, 4.93, 8.3, 16, 20, 30, 50]  # native, training, then coarse range

fig, axes = plt.subplots(1, len(target_px_sizes_nm), figsize=(3*len(target_px_sizes_nm), 3.4))

for ax, px_nm in zip(axes, target_px_sizes_nm):
    target_um = px_nm / 1000.0
    scale = ORIGINAL_PX / target_um
    new_h = max(1, round(crop.shape[0] * scale))
    new_w = max(1, round(crop.shape[1] * scale))
    down = resize(crop, (new_h, new_w), order=3, preserve_range=True,
                  anti_aliasing=(scale < 1)).astype(np.uint8)
    # upsample back to original crop size with nearest-neighbor so pixelation is visible
    disp = resize(down, crop.shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
    ax.imshow(disp, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"{px_nm:g} nm/px\n({new_w}x{new_h}px)", fontsize=10)
    ax.axis("off")

fig.suptitle(f"Same {patch_um:.0f}x{patch_um:.0f} um patch resampled to different pixel sizes\n(sub-nyuMouse07_sample-0001)", fontsize=12)
plt.tight_layout()
plt.savefig("tmp_viz/resolution_comparison.png", dpi=150, bbox_inches="tight")
print("Saved tmp_viz/resolution_comparison.png")
