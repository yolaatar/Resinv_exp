#!/usr/bin/env python3
"""
Prepare TEM1 dataset for nnUNet training — witness model (single resolution).

- Splits 20 subjects 80/20 (train/test) by subject, deterministically
- Creates nnUNet_raw/Dataset001_TEM_witness/ with imagesTr and labelsTr
- Combines axon + myelin GT into multi-class label: 0=bg, 1=axon, 2=myelin
- Saves the train/test split to subject_split.json for full reproducibility
- Test subjects are NEVER copied into nnUNet_raw (kept for final evaluation only)

Usage (on cluster):
    python prepare_dataset_witness.py \
        --data-dir ~/duke/temp/yolaatar/resinv_exp/data/TEM1 \
        --nnunet-raw ~/duke/temp/yolaatar/nnunet_resinv/nnUNet_raw

Tested with: nnunetv2==2.2.1
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

DATASET_ID = 1
DATASET_NAME = "Dataset001_TEM_witness"
TRAIN_RATIO = 0.80
SEED = 42


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
    """0=background, 1=axon, 2=myelin. Myelin overrides axon if overlap."""
    label = np.zeros((h, w), dtype=np.uint8)
    if axon_path is not None:
        label[np.array(Image.open(axon_path).convert("L")) > 0] = 1
    if myelin_path is not None:
        label[np.array(Image.open(myelin_path).convert("L")) > 0] = 2
    return label


def to_case_id(img_name: str) -> str:
    """Convert image stem to a safe nnUNet case ID (replace hyphens)."""
    return img_name.replace("-", "_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Path to TEM1 BIDS dataset root")
    parser.add_argument("--nnunet-raw", type=Path, required=True,
                        help="Path to nnUNet_raw directory")
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    dataset_dir = args.nnunet_raw / DATASET_NAME
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    # --- Subject split ---
    subjects = find_subjects(args.data_dir)
    print(f"Found {len(subjects)} subjects: {subjects}")

    rng = random.Random(args.seed)
    shuffled = subjects[:]
    rng.shuffle(shuffled)
    n_train = round(len(shuffled) * args.train_ratio)
    train_subjects = sorted(shuffled[:n_train])
    test_subjects = sorted(shuffled[n_train:])

    print(f"\nTrain ({len(train_subjects)}): {train_subjects}")
    print(f"Test  ({len(test_subjects)}):  {test_subjects}")

    split = {
        "train_subjects": train_subjects,
        "test_subjects": test_subjects,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
    }
    (dataset_dir / "subject_split.json").write_text(json.dumps(split, indent=2))
    # Also save to data dir root for easy retrieval
    (args.data_dir / "subject_split.json").write_text(json.dumps(split, indent=2))
    print(f"\nSplit saved.")

    # --- Prepare nnUNet dataset from training subjects only ---
    train_images = find_images(args.data_dir, train_subjects)
    print(f"\nPreparing {len(train_images)} training images...")

    n_ok, skipped = 0, []
    for img_path in train_images:
        img_name = img_path.stem
        axon_gt = find_gt(args.data_dir, img_name, "axon")
        myelin_gt = find_gt(args.data_dir, img_name, "myelin")

        if axon_gt is None and myelin_gt is None:
            print(f"  SKIP {img_name}: no GT")
            skipped.append(img_name)
            continue

        img = Image.open(img_path).convert("L")
        case_id = to_case_id(img_name)

        img.save(images_tr / f"{case_id}_0000.png")

        label = make_multiclass_label(axon_gt, myelin_gt, img.height, img.width)
        Image.fromarray(label).save(labels_tr / f"{case_id}.png")

        n_ok += 1
        if n_ok % 20 == 0:
            print(f"  {n_ok}/{len(train_images)} done")

    print(f"\nPrepared: {n_ok} images  |  Skipped: {len(skipped)}")

    # --- dataset.json ---
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
            "TEM1 myelinated axon/myelin segmentation. "
            "Witness model: single resolution (original acquisition pixel size). "
            "Train/test split by subject (80/20, seed=42)."
        ),
    }
    (dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))

    print(f"\nDataset ready: {dataset_dir}")
    print(f"  {n_ok} training cases")
    print(f"\nNext:")
    print(f"  export nnUNet_raw={args.nnunet_raw}")
    print(f"  nnUNetv2_plan_and_preprocess -d {DATASET_ID} -c 2d --verify_dataset_integrity")
    print(f"  CUDA_VISIBLE_DEVICES=0 nnUNetv2_train {DATASET_ID} 2d 0")


if __name__ == "__main__":
    main()
