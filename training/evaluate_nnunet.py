#!/usr/bin/env python3
"""
Run nnUNet inference on TEM1 test images at multiple pixel sizes.

Each image is downsampled to each target pixel size, segmented, then the
multi-class output (0=bg, 1=axon, 2=myelin) is split into separate PNGs.

Output structure (compatible with recompute_metrics.py):
  {output_dir}/{model_name}/{img_name}/predictions/
    {img_name}_px{px}um_seg-axon.png
    {img_name}_px{px}um_seg-myelin.png

Usage (on cluster):
  source ~/resinv_exp/venv_resinv/bin/activate
  export nnUNet_results=~/duke/temp/yolaatar/nnunet_resinv/nnUNet_results

  CUDA_VISIBLE_DEVICES=1 python evaluate_nnunet.py \\
    --model-dir ${nnUNet_results}/Dataset001_TEM_witness/nnUNetTrainer__nnUNetPlans__2d \\
    --model-name witness \\
    --data-dir ~/duke/temp/yolaatar/resinv_exp/data/TEM1 \\
    --output-dir ~/duke/temp/yolaatar/resinv_exp/results_nnunet \\
    --gpu-id 0

Tested with: nnunetv2==2.2.1
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.transform import resize

Image.MAX_IMAGE_PIXELS = None

ORIGINAL_PX_TEM1 = 0.00236  # μm/px

# Pixel sizes to evaluate: native + progressively coarser
DEFAULT_PX_SIZES = [0.00236, 0.00330, 0.00493, 0.00690, 0.01000, 0.01600]

# Multi-class label map (must match prepare_dataset_*.py)
LABEL_MAP = {1: "axon", 2: "myelin"}


def downsample(img: np.ndarray, scale: float) -> np.ndarray:
    new_h = max(1, round(img.shape[0] * scale))
    new_w = max(1, round(img.shape[1] * scale))
    out = resize(img, (new_h, new_w), order=3, preserve_range=True, anti_aliasing=True)
    return out.astype(np.uint8)


def px_tag(px: float) -> str:
    return f"px{px:.4g}um"


def find_test_images(data_dir: Path, split_path: Path) -> list[Path]:
    split = json.loads(split_path.read_text())
    test_subjects = split["test_subjects"]
    images = []
    for subject in test_subjects:
        micr = data_dir / subject / "micr"
        if not micr.exists():
            continue
        for p in sorted(micr.glob("*.png")) + sorted(micr.glob("*.tif")):
            if "_seg-" not in p.name:
                images.append(p)
    return images


def predict(predictor, img_arr: np.ndarray) -> np.ndarray:
    """
    img_arr: (H, W) uint8 grayscale — already downsampled to target pixel size.
    Returns (H, W) integer segmentation map with values 0/1/2.

    We pass spacing=[999,1,1] (matches PNG training default) so nnUNet does not
    internally resample; it processes the image at whatever size we give it.
    """
    inp = img_arr.astype(np.float32)[np.newaxis]  # (1, H, W)
    props = {"spacing": [999, 1, 1]}
    seg = predictor.predict_single_npy_array(inp, props, None, False)
    return seg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Trainer output dir (contains fold_0/ and plans.json)")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Name for this model (used as subdirectory in output-dir)")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="TEM1 dataset root (BIDS structure)")
    parser.add_argument("--output-dir", type=Path, default=Path("./results_nnunet"))
    parser.add_argument("--split-file", type=Path, default=None,
                        help="subject_split.json (default: {data-dir}/subject_split.json)")
    parser.add_argument("--px-sizes", type=float, nargs="+", default=DEFAULT_PX_SIZES,
                        help="Pixel sizes in μm/px to evaluate")
    parser.add_argument("--original-px", type=float, default=ORIGINAL_PX_TEM1,
                        help="Native pixel size of the dataset (μm/px)")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_best.pth",
                        help="Checkpoint filename (checkpoint_best.pth or checkpoint_final.pth)")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="CUDA device index (0 = first visible GPU per CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--images", type=str, nargs="*", default=None,
                        help="Specific image stems to evaluate (default: all test images)")
    args = parser.parse_args()

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    device = torch.device("cuda", args.gpu_id)
    split_path = args.split_file or (args.data_dir / "subject_split.json")

    print(f"Loading model: {args.model_dir}")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(args.model_dir),
        use_folds=(0,),
        checkpoint_name=args.checkpoint,
    )
    print("Model loaded.")

    test_images = find_test_images(args.data_dir, split_path)
    if args.images:
        test_images = [p for p in test_images if p.stem in args.images]

    print(f"\nModel     : {args.model_name}")
    print(f"Images    : {len(test_images)}")
    print(f"Px sizes  : {args.px_sizes}")
    print(f"Output    : {args.output_dir}")

    for img_path in test_images:
        img_name = img_path.stem
        print(f"\n[{img_name}]")

        img_orig = np.array(Image.open(img_path).convert("L"))
        out_dir = args.output_dir / args.model_name / img_name / "predictions"
        out_dir.mkdir(parents=True, exist_ok=True)

        for px in args.px_sizes:
            scale = args.original_px / px
            img_ds = img_orig if abs(scale - 1.0) < 1e-4 else downsample(img_orig, scale)

            seg = predict(predictor, img_ds)
            tag = px_tag(px)

            for class_id, label in LABEL_MAP.items():
                mask = (seg == class_id).astype(np.uint8) * 255
                Image.fromarray(mask).save(out_dir / f"{img_name}_{tag}_seg-{label}.png")

            print(f"  {px:.5f} μm/px  ({img_ds.shape[1]}×{img_ds.shape[0]}px)")

    print("\nDone.")


if __name__ == "__main__":
    main()
