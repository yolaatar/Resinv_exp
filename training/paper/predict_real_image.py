"""Segment one real image (no ground truth) with a paper model, optionally through canonical resampling.

    python predict_real_image.py --image IMG.png --px 0.00142 --model-dir <trainer dir> --name control_f0 \
        --out-dir ~/resinv_exp/results_paper/real/<image stem> [--fold 0] [--canonical-px 0.00236]

Same predictor settings, resampling and label map as training/evaluate_nnunet.py, so a prediction here
matches what the TEM1 sweep would produce for an image of this pixel size. Differences, both because the
image can be far larger than the TEM1 test images (237 Mpx for the 1.42 nm montage):
  - nnU-Net is called in memory (predict_single_npy_array) instead of through files: its file reader runs
    in worker processes where PIL's decompression-bomb limit (~179 Mpx) cannot be lifted.
  - PIL's limit is lifted in this process for reading the input and writing the masks.
Writes <out-dir>/<name>/seg.png (0 background, 1/2 per the model's dataset.json), one mask per class,
and meta.json (pixel sizes, shapes, timing).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.transform import resize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # training/
from evaluate_nnunet import load_label_map, resample  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--px", type=float, required=True, help="pixel size of the image (um/px)")
    ap.add_argument("--model-dir", type=Path, required=True, help="trainer dir (fold_*/, plans.json, dataset.json)")
    ap.add_argument("--name", required=True, help="output subfolder, e.g. control_f0")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--fold", default="0")
    ap.add_argument("--checkpoint", default="checkpoint_final.pth")
    ap.add_argument("--canonical-px", type=float, default=None,
                    help="resample to this pixel size (um/px) before inference and map labels back")
    ap.add_argument("--gpu-id", type=int, default=0)
    args = ap.parse_args()

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    out = args.out_dir / args.name
    if (out / "seg.png").exists():
        print(f"{out}/seg.png exists, skipping")
        return
    out.mkdir(parents=True, exist_ok=True)

    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                device=torch.device("cuda", args.gpu_id), verbose=False,
                                verbose_preprocessing=False, allow_tqdm=False)
    fold = "all" if args.fold == "all" else int(args.fold)
    predictor.initialize_from_trained_model_folder(str(args.model_dir), use_folds=(fold,),
                                                   checkpoint_name=args.checkpoint)
    label_map = load_label_map(args.model_dir)

    img = np.array(Image.open(args.image).convert("L"))
    print(f"{args.name}: image {img.shape[1]}x{img.shape[0]} at {args.px} um/px, labels {label_map}")
    t0 = time.perf_counter()
    img_input = img
    if args.canonical_px:
        scale = args.px / args.canonical_px
        img_input = resample(img, scale)
        print(f"  canonical: resampled x{scale:.4f} to {img_input.shape[1]}x{img_input.shape[0]}")
    seg = predictor.predict_single_npy_array(img_input[None, None].astype(np.float32), {"spacing": (999, 1, 1)})
    seg = np.asarray(seg)[0].astype(np.uint8)
    if seg.shape != img.shape:
        seg = resize(seg, img.shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
    seconds = time.perf_counter() - t0

    Image.fromarray(seg).save(out / "seg.png")
    for class_id, label in label_map.items():
        Image.fromarray(((seg == class_id) * 255).astype(np.uint8)).save(out / f"seg-{label}.png")
    meta = {"image": str(args.image), "px_um": args.px, "model_dir": str(args.model_dir), "fold": args.fold,
            "checkpoint": args.checkpoint, "canonical_px_um": args.canonical_px, "shape": list(img.shape),
            "segmented_shape": list(img_input.shape), "seconds": round(seconds, 1),
            "class_fractions": {label: round(float((seg == c).mean()), 4) for c, label in label_map.items()}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  done in {seconds:.0f} s: {meta['class_fractions']}")


if __name__ == "__main__":
    main()
