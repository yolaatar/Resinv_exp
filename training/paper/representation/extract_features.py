#!/usr/bin/env python3
"""
Extract per-stage nnU-Net features from TEM1 test images across pixel sizes, for the
representation analysis (E3 in the manuscript's EXPERIMENT_PLAN.md: CKA, pixel-size probe,
stage shift, UMAP).

Each test image is resampled to each pixel size exactly as in the sweep (evaluate_nnunet.resample),
normalized with the model's own preprocessor, and sent through the network in a single forward
pass (no sliding window). Forward hooks on every encoder and decoder stage capture the feature
maps, which are pooled right away into three kinds of vectors, so nothing full-resolution is kept:

  obj    one vector per GT object: each axon (connected component of the native axon mask) and
         its myelin sheath (myelin pixels assigned to the nearest axon). Pooling weights each
         feature cell by how many native pixels of the object it covers, so the same physical
         object is compared at every pixel size, with no interpolation of the features.
  grid   one vector per cell of a fixed physical grid (--grid-nm), same area-weighted pooling.
         Gives matched dense locations for the grid-based CKA check.
  global mean over the image (padding excluded), for the pixel-size probe.

Alongside each vector the script stores how many feature cells it spans (cov), so the analysis
can drop objects smaller than one cell at a given stage, and which fraction of the object lies
inside the processed region (frac, < 1 only when a fine pixel size had to be cropped).

The prediction of the same forward pass is scored with MONAI Dice (recompute_metrics.dice_monai,
prediction resized to native, as dice_*_native in results.csv). This direct pass has no sliding
window, Gaussian weighting or mirroring, so expect it within ~0.01 of the eval, not equal. A
larger gap means the features don't describe the evaluated model.

Output, per model: {out}/{model_name}/{image}.npz and {out}/{model_name}/dice.csv.
npz keys: px_um (P,), stages (S,), stride_<stage> (P,) in input pixels,
obj_class (O,) 1=axon 2=myelin, obj_axon_id (O,), obj_area_native (O,),
grid_yx (G, 2) cell centres in native px, then for each stage:
obj_<stage> (P, O, C) float16, obj_cov_<stage> (P, O), obj_frac (P, O),
grid_<stage> (P, G, C) float16, grid_cov_<stage> (P, G), grid_frac (P, G), global_<stage> (P, C).

Usage (tassan, inside a set_slot):
  source ~/resinv_exp/venv_resinv_v2/bin/activate
  python extract_features.py \\
      --model-dir ~/resinv_exp/nnunet_paper/nnUNet_results/Dataset102_TEM1_multires4/nnUNetTrainer__nnUNetPlans_ref102__2d \\
      --model-name multires_f0 --fold 0 \\
      --data-dir ~/resinv_exp/data/TEM1 --output-dir ~/resinv_exp/results_paper/representation

Tested with: nnunetv2==2.8.1
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("nnUNet_compile", "f")  # hooks need the plain module, not torch.compile's

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from skimage.segmentation import expand_labels

HERE = Path(__file__).resolve().parent
TRAINING_DIR = HERE.parent.parent
REPO_DIR = TRAINING_DIR.parent
sys.path.insert(0, str(TRAINING_DIR))
sys.path.insert(0, str(REPO_DIR))

from evaluate_nnunet import find_images, resample  # noqa: E402  same resampling as the sweep
from recompute_metrics import dice_monai, load_gt_mask, resample_mask  # noqa: E402  same scoring

Image.MAX_IMAGE_PIXELS = None

# 10 sizes of the paper grid (um/px): band edges, the multires2 dip (7, 8), sampling limit.
PX_SIZES = [0.001, 0.00174, 0.00236, 0.004, 0.007, 0.008, 0.01, 0.016, 0.0213, 0.05]

# A sheath is the myelin within this distance (native px) of its axon. 200 px = 0.47 um at
# 2.36 nm, above the thickest TEM1 sheaths, so no myelin is cut off; contested pixels go to
# the nearest axon.
SHEATH_MAX_DIST = 200


def build_objects(axon: np.ndarray, myelin: np.ndarray):
    """Instance label map on the native grid: axon k -> k, its sheath -> n_axons + k.
    Returns (labels, obj_class, obj_axon_id). Sheathless axons keep an empty sheath slot,
    dropped here so every object has at least one pixel."""
    axon_lab, n = ndimage.label(axon > 0)
    sheath = expand_labels(axon_lab, distance=SHEATH_MAX_DIST) * ((myelin > 0) & (axon_lab == 0))
    labels = axon_lab.astype(np.int32)
    labels[sheath > 0] = sheath[sheath > 0] + n

    present = np.unique(labels)
    present = present[present > 0]
    remap = np.zeros(2 * n + 1, dtype=np.int32)
    remap[present] = np.arange(1, len(present) + 1)
    labels = remap[labels]
    obj_class = np.where(present <= n, 1, 2).astype(np.int8)
    obj_axon_id = np.where(present <= n, present, present - n).astype(np.int32)
    return labels, obj_class, obj_axon_id


def build_grid(shape: tuple, cell_native: float):
    """Fixed physical grid on the native image: label map (cell k -> k + 1) and cell centres."""
    h, w = shape
    gy, gx = int(np.ceil(h / cell_native)), int(np.ceil(w / cell_native))
    rows = (np.arange(h) / cell_native).astype(np.int32)
    cols = (np.arange(w) / cell_native).astype(np.int32)
    labels = (rows[:, None] * gx + cols[None, :] + 1).astype(np.int32)
    cy, cx = np.meshgrid((np.arange(gy) + 0.5) * cell_native, (np.arange(gx) + 0.5) * cell_native,
                         indexing="ij")
    return labels, np.stack([cy.ravel(), cx.ravel()], 1).astype(np.float32)


def crop_box(h: int, w: int, max_pixels: int | None):
    """Central (y0, y1, x0, x1) on the input grid with at most max_pixels pixels, same aspect."""
    if not max_pixels or h * w <= max_pixels:
        return 0, h, 0, w
    f = np.sqrt(max_pixels / (h * w))
    ch, cw = int(h * f), int(w * f)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return y0, y0 + ch, x0, x0 + cw


class Pooler:
    """Area-weighted pooling of feature maps onto native-grid label maps.

    Native pixel (y, x) falls in input pixel (floor(y * scale) - y0, floor(x * scale) - x0) of the
    processed crop, and in feature cell (that // s) for a stage of stride s. Counting native pixels
    per (label, cell) pair gives a sparse weight matrix A, and pooled = A F / A 1. The counts are
    built once at stride 1 and merged upwards (strides are powers of 2), so each stage costs a
    pass over the previous level's pairs, not over every native pixel.
    """

    def __init__(self, labels: np.ndarray, n_labels: int, scale: float, box, device):
        y0, y1, x0, x1 = box
        hv, wv = y1 - y0, x1 - x0
        h, w = labels.shape
        iy = np.floor(np.arange(h) * scale).astype(np.int64) - y0   # native row -> crop input row
        ix = np.floor(np.arange(w) * scale).astype(np.int64) - x0
        ok_y, ok_x = (iy >= 0) & (iy < hv), (ix >= 0) & (ix < wv)
        lab = labels[ok_y][:, ok_x].ravel().astype(np.int64)
        cy = np.broadcast_to(iy[ok_y][:, None], (ok_y.sum(), ok_x.sum())).ravel()
        cx = np.broadcast_to(ix[ok_x][None, :], (ok_y.sum(), ok_x.sum())).ravel()
        m = lab > 0
        self.n, self.device, self.scale, self.hw = n_labels, device, scale, (hv, wv)
        self.levels = {1: self._merge(lab[m] - 1, cy[m], cx[m], np.ones(m.sum(), np.int64), 1)}

        area = np.bincount(labels.ravel(), minlength=n_labels + 1)[1:]
        kept = np.bincount(lab[m] - 1, minlength=n_labels)
        self.frac = np.divide(kept, area, out=np.zeros(n_labels), where=area > 0).astype(np.float32)

    def _merge(self, rows, cy, cx, counts, factor):
        """Coarsen cells by factor and sum counts of identical (label, cell) pairs."""
        cy, cx = cy // factor, cx // factor
        wc = int(cx.max()) + 1 if len(cx) else 1
        hc = int(cy.max()) + 1 if len(cy) else 1
        key = (rows * hc + cy) * wc + cx
        uniq, inv = np.unique(key, return_inverse=True)
        summed = np.bincount(inv.ravel(), weights=counts, minlength=len(uniq)).astype(np.int64)
        return uniq // (hc * wc), (uniq // wc) % hc, uniq % wc, summed

    def _level(self, stride):
        if stride not in self.levels:
            base = max(s for s in self.levels if stride % s == 0)
            self.levels[stride] = self._merge(*self.levels[base], stride // base)
        return self.levels[stride]

    def __call__(self, feat: torch.Tensor, stride: int):
        """feat: (C, Hf, Wf) on device, stride in input px. Returns pooled (n, C) float16 (NaN rows
        for objects with no pixel inside the crop) and cov (n,), the number of feature cells the
        object spans (its native pixel count / native pixels per cell)."""
        c, hf, wf = feat.shape
        rows, cy, cx, counts = self._level(stride)
        a = torch.sparse_coo_tensor(
            torch.from_numpy(np.stack([rows, cy * wf + cx])), torch.from_numpy(counts.astype(np.float32)),
            (self.n, hf * wf), device=self.device)
        summed = torch.sparse.mm(a, feat.reshape(c, -1).T.float())
        weight = np.bincount(rows, weights=counts, minlength=self.n).astype(np.float32)
        pooled = summed.cpu().numpy() / np.where(weight > 0, weight, np.nan)[:, None]
        native_per_cell = (stride / self.scale) ** 2
        return pooled.astype(np.float16), (weight / native_per_cell).astype(np.float32)


def stage_modules(network):
    """(name, module) for every encoder and decoder stage. enc0 is full resolution, enc<last> the
    bottleneck; dec0 is the deepest decoder stage, dec<last> the full-resolution one."""
    mods = [(f"enc{i}", m) for i, m in enumerate(network.encoder.stages)]
    mods += [(f"dec{i}", m) for i, m in enumerate(network.decoder.stages)]
    return mods


def load_predictor(model_dir: Path, fold: str, checkpoint: str, device):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                                device=device, verbose=False, verbose_preprocessing=False,
                                allow_tqdm=False)
    f = "all" if fold == "all" else int(fold)
    predictor.initialize_from_trained_model_folder(str(model_dir), use_folds=(f,), checkpoint_name=checkpoint)
    net = predictor.network
    net = getattr(net, "_orig_mod", net)
    net.load_state_dict(predictor.list_of_parameters[0])
    net.eval().to(device)
    return predictor, net


def normalize(predictor, img: np.ndarray) -> np.ndarray:
    """Model's own preprocessing on the whole (uncropped) image, so per-image normalization
    statistics match the eval. PNG spacing is 1, so no resampling happens here."""
    pm, cm = predictor.plans_manager, predictor.configuration_manager
    pre = cm.preprocessor_class(verbose=False)
    data = img[None, None].astype(np.float32)  # (c, 1, H, W), as NaturalImage2DIO reads it
    out, _, props = pre.run_case_npy(data, None, {"spacing": [999, 1, 1]}, pm, cm, predictor.dataset_json)
    bbox = props["bbox_used_for_cropping"]
    full = np.zeros(data.shape, dtype=np.float32)
    full[:, bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]] = out
    return full[0, 0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-dir", type=Path, required=True, help="Trainer dir (fold_*/, plans.json)")
    ap.add_argument("--model-name", required=True, help="Output subfolder, e.g. multires_f0")
    ap.add_argument("--fold", default="0")
    ap.add_argument("--checkpoint", default="checkpoint_final.pth")
    ap.add_argument("--data-dir", type=Path, required=True, help="TEM1 BIDS root")
    ap.add_argument("--split-file", type=Path, default=None, help="default: {data-dir}/subject_split.json")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--original-px", type=float, default=0.00236, help="native pixel size (um/px)")
    ap.add_argument("--px-sizes", type=float, nargs="+", default=PX_SIZES, help="um/px")
    ap.add_argument("--grid-nm", type=float, default=200.0, help="cell size of the physical grid (nm)")
    ap.add_argument("--max-input-pixels", type=int, default=6_000_000,
                    help="crop the input (centrally, after normalization) above this many pixels")
    ap.add_argument("--images", nargs="*", default=None, help="only these image stems")
    ap.add_argument("--max-images", type=int, default=None, help="first N images (quick tests)")
    ap.add_argument("--device", default="cuda", help="cuda, cuda:1, mps or cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    predictor, net = load_predictor(args.model_dir, args.fold, args.checkpoint, device)
    label_map = {v: k for k, v in predictor.dataset_json["labels"].items() if k in ("axon", "myelin")}
    stages = stage_modules(net)
    # Per-axis product of the pooling kernels, e.g. 7 pools of [2, 2] -> (128, 128)
    total_stride = np.prod(predictor.configuration_manager.pool_op_kernel_sizes, axis=0).astype(int)
    assert total_stride[0] == total_stride[1], f"anisotropic pooling {total_stride} not supported"
    total_stride = int(total_stride[0])
    print(f"Model {args.model_name}: {len(stages)} stages, total stride {total_stride}, labels {label_map}")

    captured = {}
    for name, mod in stages:
        mod.register_forward_hook(lambda m, i, o, name=name: captured.__setitem__(name, o.detach()))

    images = find_images(args.data_dir, args.split_file or args.data_dir / "subject_split.json")
    if args.images:
        images = [p for p in images if p.stem in args.images]
    images = [p for p in images if load_gt_mask(p.stem, "axon", args.data_dir) is not None]
    if args.max_images:
        images = images[:args.max_images]

    out_dir = args.output_dir / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    dice_path = out_dir / "dice.csv"
    dice_rows = []
    grid_cell_native = args.grid_nm / (args.original_px * 1000)
    use_amp = device.type == "cuda"

    for img_path in images:
        name = img_path.stem
        npz_path = out_dir / f"{name}.npz"
        if npz_path.exists():
            print(f"[{name}] skipped (done)")
            continue
        t0 = time.perf_counter()
        img = np.array(Image.open(img_path).convert("L"))
        gt = {lab: load_gt_mask(name, lab, args.data_dir) for lab in ("axon", "myelin")}
        obj_labels, obj_class, obj_axon_id = build_objects(gt["axon"], gt["myelin"])
        grid_labels, grid_yx = build_grid(img.shape, grid_cell_native)
        n_obj, n_grid = len(obj_class), len(grid_yx)

        res = {"px_um": np.array(args.px_sizes, dtype=np.float32), "obj_class": obj_class,
               "obj_axon_id": obj_axon_id, "grid_yx": grid_yx,
               "obj_area_native": np.bincount(obj_labels.ravel(), minlength=n_obj + 1)[1:].astype(np.int32),
               "obj_frac": np.zeros((len(args.px_sizes), n_obj), np.float32),
               "grid_frac": np.zeros((len(args.px_sizes), n_grid), np.float32)}

        for pi, px in enumerate(args.px_sizes):
            scale = args.original_px / px
            img_px = img if abs(scale - 1) < 1e-4 else resample(img, scale)
            norm = normalize(predictor, img_px)
            y0, y1, x0, x1 = crop_box(*norm.shape, args.max_input_pixels)
            x = norm[y0:y1, x0:x1]
            hv, wv = x.shape
            hp, wp = -(-hv // total_stride) * total_stride, -(-wv // total_stride) * total_stride
            x = np.pad(x, ((0, hp - hv), (0, wp - wv)))  # zero = mean after z-score, as nnU-Net pads
            xt = torch.from_numpy(x)[None, None].to(device)

            captured.clear()
            with torch.no_grad(), torch.autocast(device.type, enabled=use_amp):
                logits = net(xt)
            seg = logits[0].argmax(0)[:hv, :wv].cpu().numpy().astype(np.uint8)

            # Dice on the processed region, prediction resized to native as in results.csv
            ny0, ny1 = int(round(y0 / scale)), int(round(y1 / scale))
            nx0, nx1 = int(round(x0 / scale)), int(round(x1 / scale))
            row = {"image": name, "pixel_size_um": px, "input_h": img_px.shape[0], "input_w": img_px.shape[1],
                   "cropped": int((y1 - y0, x1 - x0) != img_px.shape)}
            for cid, lab in label_map.items():
                ref = gt[lab][ny0:ny1, nx0:nx1]
                pred = resample_mask((seg == cid).astype(np.uint8) * 255, ref.shape)
                row[f"dice_{lab}_native"] = dice_monai(pred, ref)
            dice_rows.append(row)

            box = (y0, y1, x0, x1)
            pool_dev = device if device.type == "cuda" else torch.device("cpu")  # no sparse mm on mps
            obj_pool = Pooler(obj_labels, n_obj, scale, box, pool_dev)
            grid_pool = Pooler(grid_labels, n_grid, scale, box, pool_dev)
            res["obj_frac"][pi], res["grid_frac"][pi] = obj_pool.frac, grid_pool.frac
            for sname, _ in stages:
                feat = captured[sname][0].to(pool_dev)
                stride = hp // feat.shape[1]
                for key, pool, n in (("obj", obj_pool, n_obj), ("grid", grid_pool, n_grid)):
                    pooled, cov = pool(feat, stride)
                    res.setdefault(f"{key}_{sname}", np.zeros((len(args.px_sizes), n, feat.shape[0]), np.float16))[pi] = pooled
                    res.setdefault(f"{key}_cov_{sname}", np.zeros((len(args.px_sizes), n), np.float32))[pi] = cov
                vh, vw = -(-hv // stride), -(-wv // stride)
                g = feat[:, :vh, :vw].float().mean((1, 2)).cpu().numpy()
                res.setdefault(f"global_{sname}", np.zeros((len(args.px_sizes), feat.shape[0]), np.float32))[pi] = g
                res.setdefault(f"stride_{sname}", np.zeros(len(args.px_sizes), np.int32))[pi] = stride
            captured.clear()
            del logits, xt
            print(f"  [{name}] {px * 1000:6.2f} nm  input {img_px.shape[1]}x{img_px.shape[0]}"
                  f"{' (cropped)' if row['cropped'] else ''}  axon {row.get('dice_axon_native', float('nan')):.3f}"
                  f"  myelin {row.get('dice_myelin_native', float('nan')):.3f}", flush=True)

        res["stages"] = np.array([s for s, _ in stages])
        np.savez_compressed(npz_path, **res)
        print(f"[{name}] {n_obj} objects, {n_grid} grid cells, {time.perf_counter() - t0:.0f}s -> {npz_path.name}")

        new = not dice_path.exists()
        with open(dice_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dice_rows[0]))
            if new:
                w.writeheader()
            w.writerows(dice_rows)
        dice_rows = []

    print("Done.")


if __name__ == "__main__":
    main()
