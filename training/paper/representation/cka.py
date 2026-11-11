#!/usr/bin/env python3
"""
CKA between nnU-Net features at different pixel sizes, from extract_features.py output (E3).

Debiased linear CKA (unbiased HSIC, Song et al. 2012; Nguyen et al. 2021), on matched rows: the
same GT object (mode obj) or the same physical grid cell (mode grid) seen at two pixel sizes.
A row is used only if it lies fully inside the processed region at both sizes (frac == 1), has no
NaN and, optionally, spans at least --min-cov feature cells at both sizes.

Kinds of comparison written to one long CSV:
  within    same model, same stage, every pair of pixel sizes            -> CKA-to-native curves, heatmaps
  shift     same model, native stage s vs stage s' at pixel size p       -> stage-shift test
  floor     as within, but rows of the second size shuffled              -> chance level
  ceiling   two models (e.g. fold 0 vs fold 1), same stage, same size    -> noise ceiling (--pairs)

Usage (tassan, CPU):
  python cka.py --feat-dir ~/resinv_exp/results_paper/representation \\
      --models control_f0 multires_f0 multires2_f0 multires_gap10_f0 \\
      --pairs control_f0:control_f1 multires_f0:multires_f1 \\
      --out ~/resinv_exp/results_paper/representation/cka_obj.csv
  # dense check: --mode grid --max-rows 5000
"""

import argparse
import csv
import glob
import os
from pathlib import Path

import numpy as np

REF_PX = 0.00236


def hsic_u(k: np.ndarray, l: np.ndarray) -> float:
    """Unbiased HSIC estimator on Gram matrices (diagonals are zeroed here)."""
    n = k.shape[0]
    k, l = k.copy(), l.copy()
    np.fill_diagonal(k, 0)
    np.fill_diagonal(l, 0)
    # tr(KL) = sum(K * L) and 1'KL1 = (K1)'(L1) for symmetric K, L: O(n^2), no n^3 product
    tr_kl = float((k * l).sum())
    one_kl_one = float(k.sum(1) @ l.sum(1))
    return (tr_kl + k.sum() * l.sum() / ((n - 1) * (n - 2)) - 2 * one_kl_one / (n - 2)) / (n * (n - 3))


def cka(x: np.ndarray, y: np.ndarray) -> float:
    """Debiased linear CKA between row-matched feature matrices (n, c1) and (n, c2)."""
    x = x - x.mean(0)
    y = y - y.mean(0)
    k, l = x @ x.T, y @ y.T
    denom = np.sqrt(max(hsic_u(k, k), 0) * max(hsic_u(l, l), 0))
    return float(hsic_u(k, l) / denom) if denom > 0 else float("nan")


class Features:
    """Lazy per-stage access to one model's npz files, rows concatenated over images."""

    def __init__(self, folder: Path, mode: str):
        self.files = sorted(glob.glob(str(folder / "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"no npz in {folder}")
        self.mode = mode
        z = np.load(self.files[0])
        self.px = z["px_um"]
        self.stages = [str(s) for s in z["stages"]]
        meta = [np.load(f) for f in self.files]
        n_rows = [len(m["obj_class"]) if mode == "obj" else len(m["grid_yx"]) for m in meta]
        self.image = np.repeat(np.arange(len(self.files)), n_rows)
        self.subject = np.repeat([os.path.basename(f).split("_")[0] for f in self.files], n_rows)
        self.cls = (np.concatenate([m["obj_class"] for m in meta]) if mode == "obj"
                    else np.zeros(sum(n_rows), np.int8))
        self.frac = np.concatenate([m[f"{mode}_frac"] for m in meta], axis=1)  # (P, N)
        self._cache = {}

    def stage(self, s: str):
        """(feat (P, N, C) float32, cov (P, N)) for one stage; keeps one stage in memory."""
        if s not in self._cache:
            self._cache.clear()
            f = np.concatenate([np.load(p)[f"{self.mode}_{s}"] for p in self.files], axis=1).astype(np.float32)
            c = np.concatenate([np.load(p)[f"{self.mode}_cov_{s}"] for p in self.files], axis=1)
            self._cache[s] = (f, c)
        return self._cache[s]

    def pi(self, px: float) -> int:
        i = int(np.argmin(np.abs(self.px - px)))
        assert abs(self.px[i] - px) < 1e-6, f"{px} not extracted"
        return i


def valid(fa, ca, frac_a, fb, cb, frac_b, min_cov):
    ok = (frac_a == 1) & (frac_b == 1) & (ca >= min_cov) & (cb >= min_cov)
    return ok & ~np.isnan(fa).any(1) & ~np.isnan(fb).any(1)


def subsample(idx: np.ndarray, max_rows: int | None, rng) -> np.ndarray:
    if max_rows and len(idx) > max_rows:
        return np.sort(rng.choice(idx, max_rows, replace=False))
    return idx


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--feat-dir", type=Path, required=True)
    ap.add_argument("--models", nargs="+", required=True, help="subfolders of --feat-dir")
    ap.add_argument("--pairs", nargs="*", default=[], help="a:b model pairs for the ceiling")
    ap.add_argument("--mode", choices=["obj", "grid"], default="obj")
    ap.add_argument("--classes", nargs="+", default=["all", "axon", "myelin"],
                    help="obj mode: rows to use (grid mode always uses all)")
    ap.add_argument("--min-cov", type=float, default=0.0,
                    help="min feature cells an object spans at both sizes (0 keeps every object)")
    ap.add_argument("--max-rows", type=int, default=None, help="random row cap per comparison (grid)")
    ap.add_argument("--no-shift", action="store_true", help="skip the stage-shift matrix")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    classes = args.classes if args.mode == "obj" else ["all"]
    cls_code = {"all": None, "axon": 1, "myelin": 2}
    fields = ["kind", "mode", "class", "model_a", "model_b", "stage_a", "stage_b", "px_a_nm", "px_b_nm", "n", "cka"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fh = open(args.out, "w", newline="")
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()

    def emit(kind, cl, ma, mb, sa, sb, pa, pb, x, y):
        w.writerow({"kind": kind, "mode": args.mode, "class": cl, "model_a": ma, "model_b": mb,
                    "stage_a": sa, "stage_b": sb, "px_a_nm": round(pa * 1000, 3), "px_b_nm": round(pb * 1000, 3),
                    "n": len(x), "cka": round(cka(x, y), 5) if len(x) > 4 else ""})

    def rows(F, cl, ia, ib, fa, ca, fb, cb):
        keep = valid(fa[ia], ca[ia], F.frac[ia], fb[ib], cb[ib], F.frac[ib], args.min_cov)
        if cls_code[cl] is not None:
            keep &= F.cls == cls_code[cl]
        return subsample(np.flatnonzero(keep), args.max_rows, rng)

    for m in args.models:
        F = Features(args.feat_dir / m, args.mode)
        ref = F.pi(REF_PX)
        print(f"{m}: {len(F.files)} images, {len(F.cls)} rows, {len(F.px)} sizes, {len(F.stages)} stages")
        for s in F.stages:
            f, c = F.stage(s)
            for cl in classes:
                for ia, pa in enumerate(F.px):
                    for ib, pb in enumerate(F.px):
                        if ib < ia:
                            continue
                        idx = rows(F, cl, ia, ib, f, c, f, c)
                        emit("within", cl, m, m, s, s, pa, pb, f[ia, idx], f[ib, idx])
                        if ia == ref and ib != ia:
                            emit("floor", cl, m, m, s, s, pa, pb, f[ia, idx], f[ib, rng.permutation(idx)])
            print(f"  {s} within/floor done", flush=True)

        if args.no_shift:
            continue
        # Stage shift: native features of stage sa against stage sb at every size. Needs two stages
        # in memory, so load the native slice of each stage once.
        native = {}
        for s in F.stages:
            f, c = F.stage(s)
            native[s] = (f[ref].copy(), c[ref].copy())
        for sb in F.stages:
            f, c = F.stage(sb)
            for ib, pb in enumerate(F.px):
                for sa in F.stages:
                    fa, ca = native[sa]
                    keep = valid(fa, ca, F.frac[ref], f[ib], c[ib], F.frac[ib], args.min_cov)
                    idx = subsample(np.flatnonzero(keep), args.max_rows, rng)
                    emit("shift", "all", m, m, sa, sb, F.px[ref], pb, fa[idx], f[ib, idx])
        print(f"  shift done", flush=True)

    for pair in args.pairs:
        ma, mb = pair.split(":")
        A, B = Features(args.feat_dir / ma, args.mode), Features(args.feat_dir / mb, args.mode)
        assert [os.path.basename(p) for p in A.files] == [os.path.basename(p) for p in B.files], "image sets differ"
        for s in A.stages:
            fa, ca = A.stage(s)
            fb, cb = B.stage(s)
            for cl in classes:
                for i, p in enumerate(A.px):
                    j = B.pi(p)
                    keep = valid(fa[i], ca[i], A.frac[i], fb[j], cb[j], B.frac[j], args.min_cov)
                    if cls_code[cl] is not None:
                        keep &= A.cls == cls_code[cl]
                    idx = subsample(np.flatnonzero(keep), args.max_rows, rng)
                    emit("ceiling", cl, ma, mb, s, s, p, p, fa[i, idx], fb[j, idx])
        print(f"{pair}: ceiling done", flush=True)

    fh.close()
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
