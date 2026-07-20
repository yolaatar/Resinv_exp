#!/usr/bin/env python3
"""
Run nnUNet inference on TEM test images at multiple pixel sizes.

Each image is resampled (up or down) to each target pixel size, segmented,
then the multi-class output (0=bg, 1=axon, 2=myelin) is split into separate PNGs.

Output structure (compatible with recompute_metrics.py):
  {output_dir}/{model_name}/{img_name}/predictions/
    {img_name}_px{px}um_seg-axon.png
    {img_name}_px{px}um_seg-myelin.png

If no subject_split.json is found, all images in the dataset are used.

Usage (on cluster):
  source ~/resinv_exp/venv_resinv/bin/activate
  export nnUNet_results=~/duke/temp/yolaatar/nnunet_resinv/nnUNet_results

  # TEM1 test split
  CUDA_VISIBLE_DEVICES=1 python evaluate_nnunet.py \\
    --model-dir ${nnUNet_results}/Dataset001_TEM_witness/nnUNetTrainer__nnUNetPlans__2d \\
    --model-name witness \\
    --data-dir ~/duke/temp/yolaatar/resinv_exp/data/TEM1 \\
    --output-dir ~/duke/temp/yolaatar/resinv_exp/results_nnunet \\
    --gpu-id 0

  # TEM2 (all images, upsampling at fine resolutions)
  CUDA_VISIBLE_DEVICES=1 python evaluate_nnunet.py \\
    --model-dir ${nnUNet_results}/Dataset001_TEM_witness/nnUNetTrainer__nnUNetPlans__2d \\
    --model-name witness \\
    --data-dir ~/duke/temp/yolaatar/resinv_exp/data/TEM2 \\
    --original-px 0.00493 \\
    --output-dir ~/duke/temp/yolaatar/resinv_exp/results_nnunet_tem2 \\
    --gpu-id 0

Tested with: nnunetv2==2.2.1
"""

import argparse
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.transform import resize

Image.MAX_IMAGE_PIXELS = None

# 22 pixel sizes: original 16 + upsampling extrapolation (1–2 nm) + coarse extrapolation (20–50 nm)
PX_SIZES = [
    # upsampling extrapolation
    0.001, 0.0015, 0.002,
    # original 16 sizes (2.36–16 nm)
    0.00236, 0.0027058, 0.0032614, 0.003931, 0.004738, 0.00493,
    0.0057108, 0.0068833, 0.0082966, 0.01, 0.0108148, 0.0116961,
    0.0126491, 0.0136798, 0.0147945, 0.016,
    # coarse extrapolation
    0.020, 0.030, 0.050,
]

# Multi-class label map (must match prepare_dataset_*.py)
LABEL_MAP = {1: "axon", 2: "myelin"}


def resample(img: np.ndarray, scale: float) -> np.ndarray:
    """Resample image by scale factor. scale<1 = downsample, scale>1 = upsample."""
    new_h = max(1, round(img.shape[0] * scale))
    new_w = max(1, round(img.shape[1] * scale))
    anti_alias = scale < 1
    out = resize(img, (new_h, new_w), order=3, preserve_range=True, anti_aliasing=anti_alias)
    return out.astype(np.uint8)


def px_tag(px: float) -> str:
    return f"px{px:.7g}um"


def find_images(data_dir: Path, split_path: Path | None) -> list[Path]:
    try:
        split_exists = split_path is not None and split_path.exists()
    except PermissionError:
        split_exists = False
    if split_exists:
        split = json.loads(split_path.read_text())
        subjects = split["test_subjects"]
        print(f"Using test split ({len(subjects)} subjects) from {split_path.name}")
    else:
        subjects = [d.name for d in sorted(data_dir.iterdir())
                    if d.is_dir() and d.name != "derivatives"]
        print(f"No split file found — using all {len(subjects)} subjects")

    images = []
    for subject in subjects:
        micr = data_dir / subject / "micr"
        if not micr.exists():
            continue
        for p in sorted(micr.glob("*.png")) + sorted(micr.glob("*.tif")):
            if "_seg-" not in p.name:
                images.append(p)
    return images


def predict(predictor, img_arr: np.ndarray) -> np.ndarray:
    """
    img_arr: (H, W) uint8 grayscale at target pixel size.
    Returns (H, W) integer seg map with values 0/1/2.

    predict_single_npy_array in v2.2.1 always writes to disk (no return mode),
    so we use predict_from_files with a temp directory instead.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        inp_dir = Path(tmpdir) / "inp"
        out_dir = Path(tmpdir) / "out"
        inp_dir.mkdir()
        out_dir.mkdir()

        Image.fromarray(img_arr).save(inp_dir / "case_0000.png")

        predictor.predict_from_files(
            [[str(inp_dir / "case_0000.png")]],
            [str(out_dir / "case")],
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
        )

        seg = np.array(Image.open(out_dir / "case.png").convert("L"))
    return seg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Trainer output dir (contains fold_0/ and plans.json)")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Name for this model (subdirectory in output-dir)")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Dataset root (BIDS structure)")
    parser.add_argument("--output-dir", type=Path, default=Path("./results_nnunet"))
    parser.add_argument("--split-file", type=Path, default=None,
                        help="subject_split.json (default: {data-dir}/subject_split.json)")
    parser.add_argument("--px-sizes", type=float, nargs="+", default=PX_SIZES,
                        help="Pixel sizes in μm/px to evaluate")
    parser.add_argument("--original-px", type=float, default=0.00236,
                        help="Native pixel size of the dataset (μm/px)")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_best.pth")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="CUDA device index (0 = first visible GPU per CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--images", type=str, nargs="*", default=None,
                        help="Specific image stems to evaluate")
    parser.add_argument("--subjects", type=str, nargs="*", default=None,
                        help="Only evaluate images from these subjects (e.g. sub-370 sub-372)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Cap number of images evaluated (random sample, seed=42)")
    args = parser.parse_args()

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    device = torch.device("cuda", args.gpu_id)
    split_path = args.split_file if args.split_file else (args.data_dir / "subject_split.json")

    print(f"Loading model: {args.model_dir}")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(args.model_dir),
        use_folds=("all",),
        checkpoint_name=args.checkpoint,
    )
    print("Model loaded.")

    images = find_images(args.data_dir, split_path)
    if args.images:
        images = [p for p in images if p.stem in args.images]
    if args.subjects:
        images = [p for p in images if p.stem.split("_")[0] in args.subjects]
    if args.max_images and len(images) > args.max_images:
        random.seed(42)
        images = sorted(random.sample(images, args.max_images))

    print(f"\nModel     : {args.model_name}")
    print(f"Images    : {len(images)}")
    print(f"Native px : {args.original_px} μm/px")
    print(f"Px sizes  : {args.px_sizes}")
    print(f"Output    : {args.output_dir}")

    for img_path in images:
        img_name = img_path.stem
        out_dir = args.output_dir / args.model_name / img_name / "predictions"

        if out_dir.exists() and all(
            (out_dir / f"{img_name}_{px_tag(px)}_seg-axon.png").exists()
            for px in args.px_sizes
        ):
            print(f"\n[{img_name}] skipped (already done)")
            continue

        print(f"\n[{img_name}]")
        img_orig = np.array(Image.open(img_path).convert("L"))
        out_dir.mkdir(parents=True, exist_ok=True)

        for px in args.px_sizes:
            scale = args.original_px / px
            img_resampled = img_orig if abs(scale - 1.0) < 1e-4 else resample(img_orig, scale)

            seg = predict(predictor, img_resampled)
            tag = px_tag(px)

            for class_id, label in LABEL_MAP.items():
                mask = (seg == class_id).astype(np.uint8) * 255
                Image.fromarray(mask).save(out_dir / f"{img_name}_{tag}_seg-{label}.png")

            arrow = "up" if scale > 1 else ("=" if abs(scale - 1.0) < 1e-4 else "down")
            print(f"  {px:.7g} μm/px  {img_resampled.shape[1]}x{img_resampled.shape[0]}  [{arrow}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
