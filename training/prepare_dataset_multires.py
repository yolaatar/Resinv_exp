#!/usr/bin/env python3
"""
Prepare TEM1 dataset for nnUNet training — multi-resolution model (model 2).

Same 80/20 subject split as the witness model (reads subject_split.json).
Each training image is duplicated at 3 additional pixel sizes:
  original (0.00236 μm/px) + 0.007 + 0.01 + 0.016 μm/px = 4x data

Images are downsampled with bicubic interpolation; GT labels with nearest-neighbor.
Each resolution version is presented as an independent training case with its own
pixel size embedded in the case ID, so nnUNet sees all resolutions as equal inputs.

Usage (on cluster):
    python prepare_dataset_multires.py \
        --data-dir ~/duke/temp/yolaatar/resinv_exp/data/TEM1 \
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

DATASET_ID = 2
DATASET_NAME = "Dataset002_TEM_multires"

# Original pixel size of TEM1 images
ORIGINAL_PX = 0.00236  # μm/px

# Additional pixel sizes to add (on top of original)
EXTRA_PX = [0.007, 0.01, 0.016]  # μm/px


def load_split(data_dir: Path) -> tuple[list[str], list[str]]:
    """Load the subject split saved by the witness dataset preparation."""
    split_path = data_dir / "subject_split.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"subject_split.json not found at {split_path}. "
            "Run prepare_dataset_witness.py first."
        )
    split = json.loads(split_path.read_text())
    return split["train_subjects"], split["test_subjects"]


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
    out = resize(img, (new_h, new_w), order=3, preserve_range=True,
                 anti_aliasing=True)
    return out.astype(np.uint8)


def downsample_label(label: np.ndarray, scale: float) -> np.ndarray:
    new_h = max(1, round(label.shape[0] * scale))
    new_w = max(1, round(label.shape[1] * scale))
    out = resize(label, (new_h, new_w), order=0, preserve_range=True,
                 anti_aliasing=False)
    return out.astype(np.uint8)


def px_tag(px: float) -> str:
    return f"px{px:.4g}um".replace(".", "p")


def to_case_id(img_name: str, px: float | None = None) -> str:
    base = img_name.replace("-", "_")
    if px is None:
        return base
    return f"{base}_{px_tag(px)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--nnunet-raw", type=Path, required=True)
    args = parser.parse_args()

    dataset_dir = args.nnunet_raw / DATASET_NAME
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    # Load same split as witness
    train_subjects, test_subjects = load_split(args.data_dir)
    print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
    print(f"Test subjects  ({len(test_subjects)}):  {test_subjects}")

    train_images = find_images(args.data_dir, train_subjects)
    all_px = [None] + EXTRA_PX  # None = original resolution
    print(f"\nResolutions: original ({ORIGINAL_PX} μm/px) + {EXTRA_PX}")
    print(f"Total cases: {len(train_images)} images × {len(all_px)} resolutions = {len(train_images) * len(all_px)}")

    n_ok, skipped = 0, []

    for img_path in train_images:
        img_name = img_path.stem
        axon_gt = find_gt(args.data_dir, img_name, "axon")
        myelin_gt = find_gt(args.data_dir, img_name, "myelin")

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
                scale = ORIGINAL_PX / px
                img_out = downsample_image(img_arr, scale)
                label_out = downsample_label(label_arr, scale)

            case_id = to_case_id(img_name, px)
            Image.fromarray(img_out).save(images_tr / f"{case_id}_0000.png")
            Image.fromarray(label_out).save(labels_tr / f"{case_id}.png")
            n_ok += 1

        if (n_ok // len(all_px)) % 20 == 0:
            print(f"  {n_ok // len(all_px)}/{len(train_images)} images done")

    print(f"\nPrepared: {n_ok} total cases ({n_ok // len(all_px)} images × {len(all_px)} resolutions)")
    print(f"Skipped: {len(skipped)}")

    dataset_json = {
        "channel_names": {"0": "TEM"},
        "labels": {
            "background": 0,
            "axon": 1,
            "myelin": 2,
        },
        "numTraining": n_ok,
        "file_ending": ".png",
        "name": DATASET_NAME,
        "description": (
            f"TEM1 multi-resolution training dataset. "
            f"Original ({ORIGINAL_PX} μm/px) + downsampled at {EXTRA_PX} μm/px. "
            f"Same 80/20 subject split as Dataset001_TEM_witness."
        ),
    }
    (dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))

    print(f"\nDataset ready: {dataset_dir}")
    print(f"\nNext:")
    print(f"  nnUNetv2_plan_and_preprocess -d {DATASET_ID} -c 2d --verify_dataset_integrity")
    print(f"  CUDA_VISIBLE_DEVICES=1 nnUNetv2_train {DATASET_ID} 2d 0")


if __name__ == "__main__":
    main()
