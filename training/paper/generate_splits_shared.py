#!/usr/bin/env python3
"""
Write splits_final.json for a paper-batch dataset: 5-fold train/validation splits made at the
SUBJECT level, identical across every dataset built from the same source images (control,
multires4, multires8, ...).

Three levels of grouping, from the outside in:
  1. Test subjects never reach nnUNet_raw (prepare_dataset_paper.py leaves them out); this
     script also checks that none slipped in.
  2. Validation folds are made of whole subjects: no animal contributes images to both the
     training and validation side of a fold, so validation scores (and the best-checkpoint
     choice made on them) are not inflated by near-duplicate tissue from the same mouse.
  3. All resolution copies of a source image (case ID suffix _px...um) belong to that image's
     subject, so they automatically stay on the same side too.

Fold assignment depends only on the subject list, the number of source images per subject
and the seed, never on the number of resolution copies. So fold k validates on the same
subjects (and the same source images) in the control and every multi-resolution dataset.

Subjects are balanced across folds by image count: shuffled with the seed, sorted by image
count (largest first), and each one given to the fold with the fewest images so far. With
TEM1 (16 training subjects of 8 images) that gives 4/3/3/3/3 subjects per validation fold.

Why not generate_splits_multires.py: it uses GroupKFold on images, which can put images of the
same subject on both sides, and balances by case count, so control and multires can end up
with different validation images.

Usage (after nnUNetv2 preprocessing, before nnUNetv2_train):
    python generate_splits_shared.py \
        --nnunet-raw ~/resinv_exp/nnunet_paper/nnUNet_raw \
        --nnunet-preprocessed /tmp/yolaatar/nnunet_preprocessed_paper \
        --dataset-name Dataset101_TEM1_control
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

RES_SUFFIX_RE = re.compile(r"_px\d+p?\d*um$")
# Case IDs are BIDS names with "-" replaced by "_" (prepare_dataset_paper.to_case_id), e.g.
# sub_nyuMouse07_sample_0001_TEM or sub_366A_sample_0001_acq_roi_TEM. BIDS subject labels are
# alphanumeric, so the subject is "sub_" plus everything up to the next underscore.
SUBJECT_RE = re.compile(r"^(sub_[A-Za-z0-9]+)_")
N_FOLDS = 5
SEED = 42


def image_group(case_id: str) -> str:
    """Source image of a case (strips the resolution suffix of multi-resolution copies)."""
    return RES_SUFFIX_RE.sub("", case_id)


def subject_of(case_id: str) -> str:
    m = SUBJECT_RE.match(case_id)
    if m is None:
        raise ValueError(f"Cannot parse a subject from case ID {case_id!r}")
    return m.group(1)


def assign_folds(images_per_subject: dict[str, int], n_folds: int, seed: int) -> dict[str, int]:
    """Deterministic, image-count-balanced assignment of whole subjects to folds."""
    order = sorted(images_per_subject)
    random.Random(seed).shuffle(order)
    # Stable sort: ties keep the shuffled order, so the result depends only on the seed.
    order.sort(key=lambda s: -images_per_subject[s])
    load = [0] * n_folds
    fold_of = {}
    for s in order:
        k = min(range(n_folds), key=lambda f: (load[f], f))
        fold_of[s] = k
        load[k] += images_per_subject[s]
    return fold_of


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnunet-raw", type=Path, required=True)
    parser.add_argument("--nnunet-preprocessed", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    raw_dir = args.nnunet_raw / args.dataset_name
    case_ids = sorted(p.name[: -len("_0000.png")] for p in (raw_dir / "imagesTr").glob("*_0000.png"))
    if not case_ids:
        raise FileNotFoundError(f"No cases found in {raw_dir / 'imagesTr'}")

    # Level 1: no test subject may appear among the training cases.
    dataset_json = json.loads((raw_dir / "dataset.json").read_text())
    test_subjects = {s.replace("-", "_") for s in dataset_json.get("resinv", {}).get("test_subjects", [])}
    leaked = sorted({subject_of(c) for c in case_ids} & test_subjects)
    if leaked:
        raise RuntimeError(f"Test subjects found among training cases: {leaked}")

    images = sorted({image_group(c) for c in case_ids})
    images_per_subject = Counter(subject_of(img) for img in images)
    subjects = sorted(images_per_subject)
    if len(subjects) < args.n_folds:
        raise ValueError(f"{len(subjects)} subjects is fewer than {args.n_folds} folds")
    fold_of = assign_folds(images_per_subject, args.n_folds, args.seed)

    splits, manifest = [], {}
    for fold in range(args.n_folds):
        val = [c for c in case_ids if fold_of[subject_of(c)] == fold]
        train = [c for c in case_ids if fold_of[subject_of(c)] != fold]
        # Levels 2 and 3: whole subjects, and therefore whole images, on one side only.
        assert not ({subject_of(c) for c in val} & {subject_of(c) for c in train})
        assert not ({image_group(c) for c in val} & {image_group(c) for c in train})
        splits.append({"train": train, "val": val})

        val_subjects = sorted(s for s in subjects if fold_of[s] == fold)
        val_images = sorted({image_group(c) for c in val})
        manifest[f"fold_{fold}"] = {"val_subjects": val_subjects, "val_images": val_images}
        res = Counter((c[len(image_group(c)):].lstrip("_") or "native") for c in val)
        print(f"Fold {fold}: val subjects {val_subjects} | {len(val_images)} val images, "
              f"{len(val)} val / {len(train)} train cases | val resolutions {dict(sorted(res.items()))}")

    out_dir = args.nnunet_preprocessed / args.dataset_name
    if not out_dir.exists():
        raise FileNotFoundError(f"{out_dir} does not exist. Preprocess the dataset first.")
    (out_dir / "splits_final.json").write_text(json.dumps(splits, indent=2))
    # Readable record of each fold's validation subjects and images: diff it between the control
    # and multires datasets to confirm they validate on the same data.
    (out_dir / "splits_val_subjects.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {out_dir / 'splits_final.json'} and splits_val_subjects.json "
          f"({len(subjects)} subjects, {len(images)} source images, seed {args.seed})")


if __name__ == "__main__":
    main()
