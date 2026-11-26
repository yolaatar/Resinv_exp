"""Build a BIDS-style test folder from the annotated images of the TEM2 release (DANDI 001350).

    python prepare_tem2_release_gt.py --release ~/resinv_exp/data/TEM2/001350 \
        --out ~/resinv_exp/data/tem2_release_gt

Only 10 of the 86 released images have manual masks (8 full 4675 x 4675 images from sub-370, 372, 374 and
375, and two ROIs from sub-373C). The release stores them as one combined mask
(<stem>_seg-axonmyelin-manual.png): 255 axon and 127 myelin in the full images, with a few edge pixels at
126, but 254 for axon in the two sub-373C ROIs (one of them saved with an alpha channel). The evaluation reads
separate axon and myelin masks, so they are split by threshold: axon >= 192, myelin 1 to 191. On the TEM2
test set (data_axondeepseg_srf_testing, values 0/127/255) this reproduces its own axon and myelin files.
Images are symlinked, subject sidecars (PixelSize 0.00493 um) copied. Nothing in the release is modified.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
AXON_MIN = 192  # axon is 254 or 255, myelin 126 or 127


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    masks = sorted((args.release / "derivatives" / "labels").glob("sub-*/micr/*_seg-axonmyelin-manual.png"))
    print(f"{len(masks)} annotated images")
    for m in masks:
        stem = m.name.replace("_seg-axonmyelin-manual.png", "")
        subject = stem.split("_")[0]
        img = args.release / subject / "micr" / f"{stem}.png"
        if not img.exists():
            raise SystemExit(f"image missing for {m.name}: {img}")
        micr = args.out / subject / "micr"
        lab = args.out / "derivatives" / "labels" / subject / "micr"
        micr.mkdir(parents=True, exist_ok=True)
        lab.mkdir(parents=True, exist_ok=True)
        link = micr / img.name
        if not link.exists():
            link.symlink_to(img.resolve())
        sidecar = args.release / subject / "micr" / f"{subject}_TEM.json"
        if sidecar.exists():
            shutil.copy2(sidecar, micr / sidecar.name)
        a = np.array(Image.open(m))
        if a.ndim == 3:  # LA: grey channel carries the labels, alpha is constant
            a = a[..., 0]
        im = Image.open(img)
        if a.shape != (im.size[1], im.size[0]):
            raise SystemExit(f"mask/image size mismatch for {stem}: {a.shape} vs {im.size}")
        axon, myelin = a >= AXON_MIN, (a > 0) & (a < AXON_MIN)
        Image.fromarray((axon * 255).astype(np.uint8)).save(lab / f"{stem}_seg-axon-manual.png")
        Image.fromarray((myelin * 255).astype(np.uint8)).save(lab / f"{stem}_seg-myelin-manual.png")
        print(f"  {stem}: {im.size[0]}x{im.size[1]}, values {np.unique(a).tolist()}, "
              f"axon {float(axon.mean()):.3f}, myelin {float(myelin.mean()):.3f}")


if __name__ == "__main__":
    main()
