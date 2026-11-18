#!/usr/bin/env python3
"""
Inference cost of segmenting a large EM image with and without the resampling step, in memory.

The paper's earlier timing went through nnU-Net's file-based predictor, where a fixed overhead of
several seconds per image hid the cost of the network itself. Here everything stays in memory and is
timed stage by stage, on a large field (a mosaic of TEM1 test images) observed at coarse pixel sizes.

Methods, for a field acquired at pixel size p:
  multires     the multi-resolution model segments the field at p as it is (no resampling)
  canonical    the field is resampled to 2.36 nm, segmented by the single-resolution model, and the
               labels are mapped back to p (nearest neighbour)
  hybrid       the field is resampled to 6.06 nm, segmented by the multi-resolution model, labels mapped
               back to p
Stages timed: resample (CPU, as in the sweep), normalize (nnU-Net preprocessor, CPU), network (nnU-Net
sliding window on the GPU, synchronised), map back (CPU). Reported: median over --repeats after one
warm-up, plus the number of pixels the network processes.

Output: {out}/cost.csv, one row per method, pixel size and repeat.

Usage (tassan, in tmux after set_slot <gpu>):
  source ~/resinv_exp/venv_resinv_v2/bin/activate
  CUDA_VISIBLE_DEVICES=1 python bench_cost.py --mosaic 2 --repeats 3
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("nnUNet_compile", "f")

import numpy as np
import torch
from PIL import Image
from skimage.transform import resize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE / "representation"))
from evaluate_nnunet import resample  # noqa: E402  same resampling as the sweep
from extract_features import normalize  # noqa: E402  model's own preprocessing, in memory

Image.MAX_IMAGE_PIXELS = None
HOME = Path.home()
RESULTS = HOME / "resinv_exp/nnunet_paper/nnUNet_results"
MODELS = {
    "control": RESULTS / "Dataset101_TEM1_control/nnUNetTrainer__nnUNetPlans_ref102__2d",
    "multires": RESULTS / "Dataset102_TEM1_multires4/nnUNetTrainer__nnUNetPlans_ref102__2d",
}
NATIVE_NM = 2.36
METHODS = {  # name: (model, target nm or None for no resampling)
    "multires": ("multires", None),
    "canonical": ("control", 2.36),
    "hybrid": ("multires", 6.06),
}


def load_predictor(model_dir, device):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                        perform_everything_on_device=True, device=device, verbose=False,
                        verbose_preprocessing=False, allow_tqdm=False)
    p.initialize_from_trained_model_folder(str(model_dir), use_folds=(0,), checkpoint_name="checkpoint_final.pth")
    return p


def mosaic(data_dir: Path, n: int) -> np.ndarray:
    """n x n tiling of TEM1 test images (native 2.36 nm), a large field of real tissue."""
    split = json.loads((data_dir / "subject_split.json").read_text())["test_subjects"]
    paths = sorted(p for s in split for p in (data_dir / s / "micr").glob("*.png") if "_seg-" not in p.name)
    tiles = [np.array(Image.open(p).convert("L")) for p in paths[:n * n]]
    h, w = tiles[0].shape
    return np.block([[tiles[r * n + c][:h, :w] for c in range(n)] for r in range(n)])


def run_once(predictor, field: np.ndarray, p_nm: float, target_nm, device):
    t = {}
    t0 = time.perf_counter()
    x = field if target_nm is None or abs(p_nm / target_nm - 1) < 1e-4 else resample(field, p_nm / target_nm)
    t["resample_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    norm = normalize(predictor, x)
    t["normalize_s"] = time.perf_counter() - t0
    data = torch.from_numpy(norm[None, None])  # (c, z, y, x) as nnU-Net's 2D data
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = predictor.predict_sliding_window_return_logits(data)
    seg = logits.argmax(0)[0]
    torch.cuda.synchronize(device)
    t["network_s"] = time.perf_counter() - t0
    seg = seg.cpu().numpy().astype(np.uint8)
    t0 = time.perf_counter()
    if seg.shape != field.shape:
        seg = resize(seg, field.shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
    t["map_back_s"] = time.perf_counter() - t0
    t["network_pixels"] = int(norm.size)
    del logits
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", type=Path, default=HOME / "resinv_exp/data/TEM1")
    ap.add_argument("--mosaic", type=int, default=2, help="n x n test images (2: 7524 x 4572 at 2.36 nm)")
    ap.add_argument("--sizes", type=float, nargs="+", default=[2.36, 4.93, 7.0, 10.0, 16.0, 21.3], help="nm")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out-dir", type=Path, default=HOME / "resinv_exp/results_paper/cost")
    args = ap.parse_args()

    device = torch.device("cuda", 0)
    predictors = {k: load_predictor(v, device) for k, v in MODELS.items()}
    native = mosaic(args.data_dir, args.mosaic)
    print(f"mosaic {native.shape[1]} x {native.shape[0]} px at {NATIVE_NM} nm "
          f"({native.shape[1] * NATIVE_NM / 1000:.1f} x {native.shape[0] * NATIVE_NM / 1000:.1f} um), "
          f"GPU {torch.cuda.get_device_name(device)}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fh = open(args.out_dir / "cost.csv", "w", newline="")
    fields = ["method", "pixel_size_nm", "repeat", "field_w", "field_h", "network_pixels",
              "resample_s", "normalize_s", "network_s", "map_back_s", "total_s"]
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    for p_nm in args.sizes:
        field = native if abs(p_nm - NATIVE_NM) < 1e-6 else resample(native, NATIVE_NM / p_nm)
        for name, (model, target) in METHODS.items():
            run_once(predictors[model], field, p_nm, target, device)  # warm-up
            for r in range(args.repeats):
                t = run_once(predictors[model], field, p_nm, target, device)
                t["total_s"] = t["resample_s"] + t["normalize_s"] + t["network_s"] + t["map_back_s"]
                w.writerow({"method": name, "pixel_size_nm": p_nm, "repeat": r, "field_w": field.shape[1],
                            "field_h": field.shape[0], **{k: (round(v, 3) if k.endswith("_s") else v)
                                                           for k, v in t.items()}})
                fh.flush()
            print(f"  {p_nm:6g} nm {name:9s} field {field.shape[1]}x{field.shape[0]}  network px "
                  f"{t['network_pixels'] / 1e6:6.1f} M  total {t['total_s']:7.2f} s "
                  f"(network {t['network_s']:.2f}, resample {t['resample_s']:.2f})", flush=True)
            torch.cuda.empty_cache()
    fh.close()
    print(f"-> {args.out_dir / 'cost.csv'}")


if __name__ == "__main__":
    main()
