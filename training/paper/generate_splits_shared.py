#!/usr/bin/env python3
"""
Write splits_final.json for a paper-batch dataset so that fold k holds out the SAME source
images in every dataset (witness, multires4, multires8, ...).

Why not generate_splits_multires.py: that one uses GroupKFold, which balances fold sizes
greedily by group size. Witness groups have 1 case and multires groups have 4, so the two
datasets can end up with different validation images for the same fold. Any witness vs
multires gap would then partly reflect which images landed in val.

Here the fold of each source image depends only on the image name and the seed: sort the
unique source images, shuffle with a fixed seed, assign fold = index % n_folds. All
resolution copies of an image (case ID suffix _px...um) follow their source image, so there
is no cross-resolution leakage either.

Usage (after nnUNetv2 preprocessing, before nnUNetv2_train):
    python generate_splits_shared.py \
        --nnunet-raw ~/resinv_exp/nnunet_paper/nnUNet_raw \
        --nnunet-preprocessed /tmp/yolaatar/nnunet_preprocessed_paper \
        --dataset-name Dataset101_TEM1_witness
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

RES_SUFFIX_RE = re.compile(r"_px\d+p?\d*um$")
N_FOLDS = 5
SEED = 42


def image_group(case_id: str) -> str:
    return RES_SUFFIX_RE.sub("", case_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnunet-raw", type=Path, required=True)
    parser.add_argument("--nnunet-preprocessed", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    images_tr = args.nnunet_raw / args.dataset_name / "imagesTr"
    case_ids = sorted(p.name[: -len("_0000.png")] for p in images_tr.glob("*_0000.png"))
    if not case_ids:
        raise FileNotFoundError(f"No cases found in {images_tr}")

    groups = sorted({image_group(c) for c in case_ids})
    order = list(groups)
    random.Random(args.seed).shuffle(order)
    fold_of = {g: i % args.n_folds for i, g in enumerate(order)}

    splits = []
    for fold in range(args.n_folds):
        val = [c for c in case_ids if fold_of[image_group(c)] == fold]
        train = [c for c in case_ids if fold_of[image_group(c)] != fold]
        assert not ({image_group(c) for c in val} & {image_group(c) for c in train})
        splits.append({"train": train, "val": val})
        res = Counter((c[len(image_group(c)):].lstrip("_") or "native") for c in val)
        print(f"Fold {fold}: {len(train)} train / {len(val)} val cases, "
              f"{len({image_group(c) for c in val})} val images, val resolutions {dict(sorted(res.items()))}")

    out_dir = args.nnunet_preprocessed / args.dataset_name
    if not out_dir.exists():
        raise FileNotFoundError(f"{out_dir} does not exist. Preprocess the dataset first.")
    (out_dir / "splits_final.json").write_text(json.dumps(splits, indent=2))

    # Human-readable record of which source images each fold validates on, so the
    # "same val images across datasets" property can be checked with a plain diff.
    manifest = {f"fold_{k}": sorted(g for g in groups if fold_of[g] == k) for k in range(args.n_folds)}
    (out_dir / "splits_val_images.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {out_dir / 'splits_final.json'} and splits_val_images.json "
          f"({len(groups)} source images, seed {args.seed})")


if __name__ == "__main__":
    main()
