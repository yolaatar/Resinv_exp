#!/usr/bin/env python3
"""
Prepare TEM1 + TEM2 dataset for nnUNet training — full multi-resolution model.

Differences from prepare_dataset_multires.py (model 2):
  - Uses 100% of TEM1 subjects (no train/test split)
  - Also includes all annotated TEM2 subjects
  - Case IDs are prefixed with t1_ / t2_ to avoid collisions

Each image from both datasets is duplicated at 3 additional pixel sizes:
  TEM1 original (0.00236 μm/px) + 0.007 + 0.01 + 0.016 μm/px
  TEM2 original (0.00493 μm/px) + 0.007 + 0.01 + 0.016 μm/px

Usage (on cluster):
    python prepare_dataset_multires_full.py \
        --tem1-dir ~/duke/temp/yolaatar/resinv_exp/data/TEM1 \
        --tem2-dir ~/duke/temp/yolaatar/resinv_exp/data/TEM2/001350 \
        --nnunet-raw ~/duke/temp/yolaatar/nnunet_resinv/nnUNet_raw

Tested with: nnunetv2==2.2.1
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.transform import resize

Image.MAX_IMAGE_PIXELS = None

DATASET_ID = 5
DATASET_NAME = "Dataset005_TEM12_multires"

TEM1_ORIGINAL_PX = 0.00236  # μm/px
TEM2_ORIGINAL_PX = 0.00493  # μm/px

# Additional pixel sizes to add for both datasets (all coarser than either native)
EXTRA_PX = [0.007, 0.01, 0.016]  # μm/px


def find_subjects(data_dir: Path) -> list[str]:
    return sorted(d.name for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("sub-"))


def find_images(data_dir: Path, subjects: list[str]) -> list[Path]:
    images = []
    for subject in subjects:
        micr = data_dir / subject / "micr"
        if not micr.exists():
            continue
        for p in sorted(micr.glob("*.png")) + sorted(micr.glob("*.tif")):
            if "_seg-" not in p.name:
                images.append(p)
    return images


def find_gt(data_dir: Path, img_name: str, label: str) -> Path | None:
    subject = img_name.split("_")[0]
    p = data_dir / "derivatives" / "labels" / subject / "micr" / f"{img_name}_seg-{label}-manual.png"
    return p if p.exists() else None


def make_multiclass_label(axon_path: Path | None, myelin_path: Path | None,
                           h: int, w: int) -> np.ndarray:
    label = np.zeros((h, w), dtype=np.uint8)
    if axon_path is not None:
        label[np.array(Image.open(axon_path).convert("L")) > 0] = 1
    if myelin_path is not None:
        label[np.array(Image.open(myelin_path).convert("L")) > 0] = 2
    return label


def downsample_image(img: np.ndarray, scale: float) -> np.ndarray:
    new_h = max(1, round(img.shape[0] * scale))
    new_w = max(1, round(img.shape[1] * scale))
    out = resize(img, (new_h, new_w), order=3, preserve_range=True, anti_aliasing=True)
    return out.astype(np.uint8)


def downsample_label(label: np.ndarray, scale: float) -> np.ndarray:
    new_h = max(1, round(label.shape[0] * scale))
    new_w = max(1, round(label.shape[1] * scale))
    out = resize(label, (new_h, new_w), order=0, preserve_range=True, anti_aliasing=False)
    return out.astype(np.uint8)


def px_tag(px: float) -> str:
    return f"px{px:.4g}um".replace(".", "p")


def to_case_id(prefix: str, img_name: str, px: float | None = None) -> str:
    base = f"{prefix}_{img_name}".replace("-", "_")
    if px is None:
        return base
    return f"{base}_{px_tag(px)}"


def process_dataset(data_dir: Path, original_px: float, prefix: str,
                    images_tr: Path, labels_tr: Path) -> tuple[int, list[str]]:
    subjects = find_subjects(data_dir)
    print(f"\n[{prefix}] Found {len(subjects)} subjects: {subjects}")

    images = find_images(data_dir, subjects)
    all_px = [None] + EXTRA_PX
    print(f"[{prefix}] {len(images)} images x {len(all_px)} resolutions = {len(images) * len(all_px)} cases")

    n_ok, skipped = 0, []

    for img_path in images:
        img_name = img_path.stem
        axon_gt = find_gt(data_dir, img_name, "axon")
        myelin_gt = find_gt(data_dir, img_name, "myelin")

        if axon_gt is None and myelin_gt is None:
            print(f"  SKIP {img_name}: no GT")
            skipped.append(img_name)
            continue

        img_arr = np.array(Image.open(img_path).convert("L"))
        label_arr = make_multiclass_label(axon_gt, myelin_gt, img_arr.shape[0], img_arr.shape[1])

        for px in all_px:
            if px is None:
                img_out = img_arr
                label_out = label_arr
            else:
                scale = original_px / px
                img_out = downsample_image(img_arr, scale)
                label_out = downsample_label(label_arr, scale)

            case_id = to_case_id(prefix, img_name, px)
            Image.fromarray(img_out).save(images_tr / f"{case_id}_0000.png")
            Image.fromarray(label_out).save(labels_tr / f"{case_id}.png")
            n_ok += 1

        done = n_ok // len(all_px)
        if done % 20 == 0:
            print(f"  [{prefix}] {done}/{len(images)} images done")

    return n_ok, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tem1-dir", type=Path, required=True,
                        help="Path to TEM1 BIDS dataset root")
    parser.add_argument("--tem2-dir", type=Path, required=True,
                        help="Path to TEM2 BIDS dataset root (001350 subdirectory)")
    parser.add_argument("--nnunet-raw", type=Path, required=True)
    args = parser.parse_args()

    dataset_dir = args.nnunet_raw / DATASET_NAME
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    print("======================================================")
    print(f" Full multi-resolution dataset preparation")
    print(f" TEM1: {args.tem1_dir}  (native {TEM1_ORIGINAL_PX} μm/px, 100% subjects)")
    print(f" TEM2: {args.tem2_dir}  (native {TEM2_ORIGINAL_PX} μm/px, all annotated subjects)")
    print(f" Extra resolutions: {EXTRA_PX} μm/px")
    print("======================================================")

    n_tem1, skip_tem1 = process_dataset(
        args.tem1_dir, TEM1_ORIGINAL_PX, "t1", images_tr, labels_tr
    )
    n_tem2, skip_tem2 = process_dataset(
        args.tem2_dir, TEM2_ORIGINAL_PX, "t2", images_tr, labels_tr
    )

    n_total = n_tem1 + n_tem2
    print(f"\nTEM1: {n_tem1} cases ({len(skip_tem1)} images skipped)")
    print(f"TEM2: {n_tem2} cases ({len(skip_tem2)} images skipped)")
    print(f"Total: {n_total} cases")

    dataset_json = {
        "channel_names": {"0": "TEM"},
        "labels": {
            "background": 0,
            "axon": 1,
            "myelin": 2,
        },
        "numTraining": n_total,
        "file_ending": ".png",
        "name": DATASET_NAME,
        "description": (
            f"TEM1 (100%) + TEM2 (all annotated subjects) multi-resolution training dataset. "
            f"TEM1 native {TEM1_ORIGINAL_PX} μm/px, TEM2 native {TEM2_ORIGINAL_PX} μm/px. "
            f"Each image duplicated at extra resolutions {EXTRA_PX} μm/px. "
            f"No held-out test split — both datasets used fully for training."
        ),
    }
    (dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))

    print(f"\nDataset ready: {dataset_dir}")
    print(f"\nNext:")
    print(f"  nnUNetv2_plan_and_preprocess -d {DATASET_ID} -c 2d --verify_dataset_integrity")
    print(f"  CUDA_VISIBLE_DEVICES=1 nnUNetv2_train {DATASET_ID} 2d 0")


if __name__ == "__main__":
    main()
