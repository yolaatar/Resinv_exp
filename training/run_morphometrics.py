#!/usr/bin/env python3
"""
Run ADS's official `axondeepseg_morphometrics` CLI on the Armand uaxon model's
predictions (results_armand_uaxon/armand_uaxon/{image}/predictions/), across
all pixel sizes, and aggregate the per-image morphometrics into one summary CSV.

Two things are missing from the raw evaluate_nnunet.py output that ADS's CLI
needs to auto-detect targets in a folder:
  1. a raw image file at each pixel size (evaluate_nnunet.py never saved the
     resampled input, only the predicted masks)
  2. a combined axonmyelin mask (axon=255, myelin=127) — ADS uses this for the
     index-overlay image it generates alongside the morphometrics file

Step 1 (prep) regenerates both by resampling the same raw source image the
same way evaluate_nnunet.py did (same resample() function, same original px),
and by compositing axon+myelin masks together.

Step 2 (run) shells out to the real `axondeepseg_morphometrics` CLI once per
pixel size (pixel size can't be mixed within a single CLI call), once for
myelinated axons (axon+myelin, produces gratio/diam/etc.) and once for
unmyelinated axons (-u flag, produces uaxon diam/area only, no gratio).

Step 3 (aggregate) reads all the per-image .xlsx files the CLI wrote and
collapses them into one results CSV: one row per (image, pixel_size), with
mean axon_diam/gratio/myelin_thickness/uaxon_diam and axon/uaxon counts.

Usage (on tassan, after activating the venv that has AxonDeepSeg installed):
    python run_morphometrics.py --results-dir ~/resinv_exp/results_armand_uaxon/armand_uaxon \\
        --data-dir ~/resinv_exp/data/testset_armand_uaxon --original-px 0.00493
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import resize

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_nnunet import load_original_px  # noqa: E402 -- reuse the same per-image
# sidecar-JSON pixel-size lookup as evaluate_nnunet.py, so datasets with mixed
# native pixel sizes (e.g. TEM3) resample raw images by the correct per-image
# value instead of one dataset-wide --original-px.

Image.MAX_IMAGE_PIXELS = None


def resample(img: np.ndarray, scale: float) -> np.ndarray:
    new_h = max(1, round(img.shape[0] * scale))
    new_w = max(1, round(img.shape[1] * scale))
    anti_alias = scale < 1
    out = resize(img, (new_h, new_w), order=3, preserve_range=True, anti_aliasing=anti_alias)
    return out.astype(np.uint8)


def px_tag(px: float) -> str:
    return f"px{px:.7g}um"


def parse_px(filename: str) -> float | None:
    m = re.search(r"_px([\d.]+)um", filename)
    return float(m.group(1)) if m else None


def load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


def prep_inputs(results_dir: Path, data_dir: Path, original_px_fallback: float) -> dict[float, list[Path]]:
    """Regenerate raw-image + axonmyelin files; return {px: [stem_paths]}."""
    px_to_stems: dict[float, list[Path]] = {}

    img_dirs = sorted(d for d in results_dir.iterdir() if d.is_dir())
    for img_dir in img_dirs:
        pred_dir = img_dir / "predictions"
        if not pred_dir.exists():
            continue
        img_name = img_dir.name
        subject = img_name.split("_")[0]
        raw_src = data_dir / subject / "micr" / f"{img_name}.png"
        if not raw_src.exists():
            print(f"  WARNING: no raw source image for {img_name} at {raw_src}, skipping")
            continue
        raw_native = load_gray(raw_src)
        original_px = load_original_px(raw_src, original_px_fallback)

        axon_files = sorted(pred_dir.glob("*_seg-axon.png"))
        px_set = sorted({parse_px(p.name) for p in axon_files if parse_px(p.name) is not None})

        for px in px_set:
            tag = px_tag(px)
            stem_path = pred_dir / f"{img_name}_{tag}.png"
            axon_path = pred_dir / f"{img_name}_{tag}_seg-axon.png"
            myelin_path = pred_dir / f"{img_name}_{tag}_seg-myelin.png"
            axonmyelin_path = pred_dir / f"{img_name}_{tag}_seg-axonmyelin.png"

            if not stem_path.exists():
                scale = original_px / px
                raw_resampled = raw_native if abs(scale - 1.0) < 1e-4 else resample(raw_native, scale)
                Image.fromarray(raw_resampled).save(stem_path)

            if not axonmyelin_path.exists() and axon_path.exists() and myelin_path.exists():
                axon_mask = load_gray(axon_path) > 0
                myelin_mask = load_gray(myelin_path) > 0
                combined = np.zeros(axon_mask.shape, dtype=np.uint8)
                combined[myelin_mask] = 127
                combined[axon_mask] = 255
                Image.fromarray(combined).save(axonmyelin_path)

            px_to_stems.setdefault(px, []).append(stem_path)

        print(f"[{img_name}] prepped {len(px_set)} pixel sizes")

    return px_to_stems


def run_cli(px_to_stems: dict[float, list[Path]], cli_bin: str) -> None:
    for px in sorted(px_to_stems):
        stems = px_to_stems[px]
        stem_args = [str(p) for p in stems]

        print(f"\n=== px={px:.7g} um  ({len(stems)} images)  myelinated ===")
        subprocess.run(
            [cli_bin, "-i", *stem_args, "-s", str(px), "-a", "circle"],
            check=True,
        )

        print(f"=== px={px:.7g} um  ({len(stems)} images)  unmyelinated ===")
        subprocess.run(
            [cli_bin, "-i", *stem_args, "-s", str(px), "-a", "circle", "-u"],
            check=True,
        )


def aggregate(results_dir: Path, model_name: str) -> pd.DataFrame:
    rows = []
    img_dirs = sorted(d for d in results_dir.iterdir() if d.is_dir())
    for img_dir in img_dirs:
        pred_dir = img_dir / "predictions"
        if not pred_dir.exists():
            continue
        img_name = img_dir.name

        for xlsx_path in sorted(pred_dir.glob(f"{img_name}_px*_axon_morphometrics.xlsx")):
            px = parse_px(xlsx_path.name)
            if px is None:
                continue
            df = pd.read_excel(xlsx_path)
            row = {
                "image": img_name,
                "model": model_name,
                "pixel_size_um": px,
                "n_axons": len(df),
            }
            for col, out_name in [
                ("gratio", "mean_gratio"),
                ("axon_diam (um)", "mean_axon_diam_um"),
                ("myelin_thickness (um)", "mean_myelin_thickness_um"),
                ("axon_area (um^2)", "mean_axon_area_um2"),
            ]:
                row[out_name] = df[col].mean() if col in df.columns and len(df) else float("nan")
            rows.append(row)

        for xlsx_path in sorted(pred_dir.glob(f"{img_name}_px*_uaxon_morphometrics.xlsx")):
            px = parse_px(xlsx_path.name)
            if px is None:
                continue
            df = pd.read_excel(xlsx_path)
            match = next((r for r in rows if r["image"] == img_name and r["pixel_size_um"] == px), None)
            if match is None:
                match = {"image": img_name, "model": model_name, "pixel_size_um": px}
                rows.append(match)
            match["n_uaxons"] = len(df)
            match["mean_uaxon_diam_um"] = df["axon_diam (um)"].mean() if "axon_diam (um)" in df.columns and len(df) else float("nan")
            match["mean_uaxon_area_um2"] = df["axon_area (um^2)"].mean() if "axon_area (um^2)" in df.columns and len(df) else float("nan")

    return pd.DataFrame(rows).sort_values(["image", "pixel_size_um"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True,
                         help="e.g. ~/resinv_exp/results_armand_uaxon/armand_uaxon")
    parser.add_argument("--data-dir", type=Path, required=True,
                         help="Raw testset dir, e.g. ~/resinv_exp/data/testset_armand_uaxon")
    parser.add_argument("--original-px", type=float, default=0.00493,
                         help="Fallback native pixel size (um/px); per-image sidecar JSON is used when present")
    parser.add_argument("--model-name", type=str, default="armand_uaxon")
    parser.add_argument("--cli-bin", type=str, default="axondeepseg_morphometrics")
    parser.add_argument("--skip-prep", action="store_true", help="Skip regenerating raw/axonmyelin files")
    parser.add_argument("--skip-run", action="store_true", help="Skip CLI calls, only aggregate existing xlsx")
    args = parser.parse_args()

    if args.skip_run:
        px_to_stems = {}
    elif args.skip_prep:
        # rebuild px_to_stems from existing files without regenerating anything
        px_to_stems = {}
        for img_dir in sorted(d for d in args.results_dir.iterdir() if d.is_dir()):
            pred_dir = img_dir / "predictions"
            for p in sorted(pred_dir.glob(f"{img_dir.name}_px*.png")):
                if "_seg-" in p.name:
                    continue
                px = parse_px(p.name)
                if px is not None:
                    px_to_stems.setdefault(px, []).append(p)
    else:
        print("=== Step 1: prepping raw images + axonmyelin masks ===")
        px_to_stems = prep_inputs(args.results_dir, args.data_dir, args.original_px)

    if not args.skip_run:
        print("\n=== Step 2: running axondeepseg_morphometrics CLI ===")
        run_cli(px_to_stems, args.cli_bin)

    print("\n=== Step 3: aggregating results ===")
    summary = aggregate(args.results_dir, args.model_name)
    out_path = args.results_dir / "morphometrics_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  ({len(summary)} rows)")


if __name__ == "__main__":
    main()
