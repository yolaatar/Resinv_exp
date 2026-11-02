#!/usr/bin/env python3
"""
Generate a group-aware splits_final.json for the TEM1 multi-resolution dataset (v2).

Why: nnUNet's default split randomly partitions the *case* list into 5 folds.
Each source image in the multires dataset is duplicated at 4 pixel sizes
(original + EXTRA_PX), so a random per-case split can put e.g. the original-res
copy of an image in train and its 0.007um/px copy in val for fold_0 — that's
leakage (near-identical content on both sides) and can also leave val with an
unbalanced mix of resolutions.

Fix: group all resolution-duplicates of the same source image together before
splitting (GroupKFold), so a whole image (all its resolution copies) goes
entirely to train or entirely to val. Since every image contributes exactly
one case per resolution, grouping by image also gives proportional resolution
balance in both splits for free.

Only fold_0 is used for training the TEM1 multires v2 model, but all 5 folds
are generated for consistency with nnUNet's expected splits_final.json format.

Usage:
    python generate_splits_multires.py \
        --nnunet-raw ~/duke/temp/yolaatar/nnunet_resinv_v2/nnUNet_raw \
        --nnunet-preprocessed /tmp/yolaatar/nnunet_preprocessed_v2 \
        --dataset-name Dataset003_TEM_multires_v2

Run after `nnUNetv2_plan_and_preprocess` (needs the preprocessed dataset dir
to exist) and before `nnUNetv2_train ... 0`.
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from sklearn.model_selection import GroupKFold

RES_SUFFIX_RE = re.compile(r"_px\d+p?\d*um$")

N_FOLDS = 5
SEED = 42


def case_ids_from_raw(nnunet_raw: Path, dataset_name: str) -> list[str]:
    images_tr = nnunet_raw / dataset_name / "imagesTr"
    case_ids = sorted(p.name[: -len("_0000.png")] for p in images_tr.glob("*_0000.png"))
    if not case_ids:
        raise FileNotFoundError(f"No cases found in {images_tr}")
    return case_ids


def image_group(case_id: str) -> str:
    """Strip the resolution suffix so all pixel-size copies of one image share a group."""
    return RES_SUFFIX_RE.sub("", case_id)


def resolution_tag(case_id: str, group: str) -> str:
    suffix = case_id[len(group):]
    return suffix.lstrip("_") if suffix else "original"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnunet-raw", type=Path, required=True)
    parser.add_argument("--nnunet-preprocessed", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True,
                        help="e.g. Dataset003_TEM_multires_v2")
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    case_ids = case_ids_from_raw(args.nnunet_raw, args.dataset_name)
    groups = [image_group(cid) for cid in case_ids]
    n_groups = len(set(groups))
    print(f"Found {len(case_ids)} cases across {n_groups} source images")

    # Shuffle group order deterministically so GroupKFold's fold sizes aren't
    # biased by however find_images() happened to sort subjects/images.
    rng = random.Random(args.seed)
    unique_groups = sorted(set(groups))
    rng.shuffle(unique_groups)
    group_rank = {g: i for i, g in enumerate(unique_groups)}
    order = sorted(range(len(case_ids)), key=lambda i: group_rank[groups[i]])
    case_ids = [case_ids[i] for i in order]
    groups = [groups[i] for i in order]

    gkf = GroupKFold(n_splits=args.n_folds)
    splits = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(case_ids, groups=groups)):
        train_cases = [case_ids[i] for i in train_idx]
        val_cases = [case_ids[i] for i in val_idx]
        splits.append({"train": train_cases, "val": val_cases})

        train_res = Counter(resolution_tag(case_ids[i], groups[i]) for i in train_idx)
        val_res = Counter(resolution_tag(case_ids[i], groups[i]) for i in val_idx)
        train_groups = len(set(groups[i] for i in train_idx))
        val_groups = len(set(groups[i] for i in val_idx))
        print(f"\nFold {fold}: {len(train_cases)} train cases ({train_groups} images), "
              f"{len(val_cases)} val cases ({val_groups} images)")
        print(f"  train resolutions: {dict(sorted(train_res.items()))}")
        print(f"  val   resolutions: {dict(sorted(val_res.items()))}")

    # Sanity check: no image group should appear in both train and val of the same fold
    for fold, split in enumerate(splits):
        train_groups = {image_group(c) for c in split["train"]}
        val_groups = {image_group(c) for c in split["val"]}
        overlap = train_groups & val_groups
        assert not overlap, f"Fold {fold}: leakage, groups in both train and val: {overlap}"
    print("\nNo cross-resolution leakage in any fold.")

    out_dir = args.nnunet_preprocessed / args.dataset_name
    if not out_dir.exists():
        raise FileNotFoundError(
            f"{out_dir} does not exist. Run nnUNetv2_plan_and_preprocess first."
        )
    out_path = out_dir / "splits_final.json"
    out_path.write_text(json.dumps(splits, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
