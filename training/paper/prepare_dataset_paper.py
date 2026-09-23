#!/usr/bin/env python3
"""
Prepare an nnUNet raw dataset for the paper retrain batch (nnUNet 2.8.1).

One script for every training-set variant, so all of them are built exactly the same way:
same subject split, same interpolation, same case naming. It generalizes
prepare_dataset_multires_v2.py: pass no --extra-px for a single-resolution (witness) dataset,
or a list of coarser pixel sizes for a multi-resolution one.

Each training image is kept at its native pixel size, plus one downsampled copy per
--extra-px value. Images are downsampled bicubic with anti-aliasing, labels nearest-neighbor
(identical to the v2 recipe). Every copy is an independent nnUNet case whose ID carries the
pixel size (e.g. ..._px0p007um), which generate_splits_shared.py relies on to keep all copies
of one source image on the same side of the train/val split.

Only downsampling is supported: every --extra-px must be coarser than --original-px.

Usage (on tassan):
    python prepare_dataset_paper.py \
        --data-dir ~/resinv_exp/data/TEM1 \
        --nnunet-raw ~/resinv_exp/nnunet_paper/nnUNet_raw \
        --dataset-id 102 --dataset-name Dataset102_TEM1_multires4 \
        --extra-px 0.007 0.01 0.016

Tested with: nnunetv2==2.8.1
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.transform import resize

Image.MAX_IMAGE_PIXELS = None


def load_split(split_file: Path) -> tuple[list[str], list[str]]:
    if not split_file.exists():
        raise FileNotFoundError(
            f"{split_file} not found. For TEM1 this is written by prepare_dataset_witness.py "
            "(seed 42); reuse it, never regenerate it, so every model shares the same test subjects."
        )
    split = json.loads(split_file.read_text())
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
    """0=background, 1=axon, 2=myelin. Myelin overrides axon on overlap."""
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


def to_case_id(img_name: str, px: float | None = None) -> str:
    base = img_name.replace("-", "_")
    return base if px is None else f"{base}_{px_tag(px)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--nnunet-raw", type=Path, required=True)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--dataset-name", required=True, help="e.g. Dataset101_TEM1_witness")
    parser.add_argument("--original-px", type=float, default=0.00236,
                        help="Native pixel size of the source dataset in um/px (TEM1: 0.00236)")
    parser.add_argument("--extra-px", type=float, nargs="*", default=[],
                        help="Extra (coarser) pixel sizes in um/px. Empty = single resolution")
    parser.add_argument("--split-file", type=Path, default=None,
                        help="Subject split JSON (default: <data-dir>/subject_split.json)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild even if the dataset folder already has cases")
    args = parser.parse_args()

    if not args.dataset_name.startswith(f"Dataset{args.dataset_id:03d}_"):
        parser.error(f"--dataset-name must start with Dataset{args.dataset_id:03d}_")
    bad = [px for px in args.extra_px if px <= args.original_px]
    if bad:
        parser.error(f"--extra-px must all be coarser than --original-px ({args.original_px}): {bad}")

    dataset_dir = args.nnunet_raw / args.dataset_name
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    if (dataset_dir / "dataset.json").exists() and not args.overwrite:
        print(f"{dataset_dir} already prepared, skipping (pass --overwrite to rebuild)")
        return
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    split_file = args.split_file or (args.data_dir / "subject_split.json")
    train_subjects, test_subjects = load_split(split_file)
    print(f"Split file: {split_file}")
    print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
    print(f"Test subjects  ({len(test_subjects)}), never written: {test_subjects}")

    train_images = find_images(args.data_dir, train_subjects)
    all_px = [None] + sorted(args.extra_px)  # None = native resolution
    print(f"\nResolutions: native ({args.original_px} um/px) + {sorted(args.extra_px)}")

    n_ok, n_images, skipped = 0, 0, []
    for img_path in train_images:
        img_name = img_path.stem
        axon_gt = find_gt(args.data_dir, img_name, "axon")
        myelin_gt = find_gt(args.data_dir, img_name, "myelin")
        if axon_gt is None and myelin_gt is None:
            skipped.append(img_name)
            continue

        img_arr = np.array(Image.open(img_path).convert("L"))
        label_arr = make_multiclass_label(axon_gt, myelin_gt, *img_arr.shape[:2])

        for px in all_px:
            if px is None:
                img_out, label_out = img_arr, label_arr
            else:
                scale = args.original_px / px
                img_out = downsample_image(img_arr, scale)
                label_out = downsample_label(label_arr, scale)
            case_id = to_case_id(img_name, px)
            Image.fromarray(img_out).save(images_tr / f"{case_id}_0000.png")
            Image.fromarray(label_out).save(labels_tr / f"{case_id}.png")
            n_ok += 1

        n_images += 1
        if n_images % 20 == 0:
            print(f"  {n_images}/{len(train_images)} images done")

    print(f"\nPrepared: {n_ok} cases ({n_images} images x {len(all_px)} resolutions)")
    print(f"Skipped (no GT): {len(skipped)} {skipped if skipped else ''}")

    dataset_json = {
        "channel_names": {"0": "TEM"},
        "labels": {"background": 0, "axon": 1, "myelin": 2},
        "numTraining": n_ok,
        "file_ending": ".png",
        "name": args.dataset_name,
        "description": (
            f"ResInv paper retrain batch (nnUNet 2.8.1). Source: {args.data_dir.name}, native "
            f"{args.original_px} um/px, extra pixel sizes {sorted(args.extra_px)} um/px. "
            f"Subject split from {split_file.name}. Splits: generate_splits_shared.py."
        ),
        # Not read by nnUNet; kept so every trained model records exactly what it saw.
        "resinv": {
            "source_dir": str(args.data_dir),
            "original_px_um": args.original_px,
            "extra_px_um": sorted(args.extra_px),
            "n_source_images": n_images,
            "train_subjects": train_subjects,
            "test_subjects": test_subjects,
        },
    }
    (dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    print(f"\nDataset ready: {dataset_dir}")


if __name__ == "__main__":
    main()
