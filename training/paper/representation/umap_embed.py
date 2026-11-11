#!/usr/bin/env python3
"""
UMAP of per-object features across pixel sizes, with scores that don't depend on the embedding (E3, D).

For each model and stage, the same objects (GT axons and sheaths fully inside the processed region at
every selected size) are taken at every selected pixel size, standardized, reduced to 50 PCA
components and embedded with UMAP. Points are object x pixel size, so an invariant representation
puts the copies of one object at different sizes next to each other, and a size-driven one groups
different objects by size.

The plot is only an illustration. The scores are computed in the 50-component PCA space, not in the
UMAP plane:
  knn_px_acc        k-NN (k = --k, self excluded) accuracy predicting the pixel size; chance 1 / sizes.
                    Low = sizes mixed (invariant, can fall below chance since an object's own copies
                    are all at other sizes); high = clustered by size.
  knn_px_oct        mean |log2| distance, in octaves, between a point's size and its neighbours'.
                    Finer than accuracy, since 7 and 8 nm are near neighbours anyway.
  knn_same_obj      fraction of neighbours that are the same object at another size; at most
                    (sizes - 1) / k (0.47 with 8 sizes, k = 15).
  knn_class_acc     k-NN accuracy predicting axon vs sheath; chance ~0.5.
  sil_px, sil_class silhouette of the size and class labels (sampled).
Scores are written for every selected size and for --band (the multires robust band by default).

Usage (tassan, CPU; pip install umap-learn first):
  python umap_embed.py --feat-dir ~/resinv_exp/results_paper/representation \\
      --models control_f0 multires_f0 multires2_f0 --stages enc5 dec3 \\
      --out-dir ~/resinv_exp/results_paper/representation/umap
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from cka import Features

PX_DEFAULT = [0.00236, 0.004, 0.007, 0.008, 0.01, 0.016, 0.0213, 0.05]


def scores(z, px, cls, obj, k):
    """k-NN and silhouette scores in the reduced feature space z (n, d)."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(z)
    idx = nn.kneighbors(z, return_distance=False)[:, 1:]  # drop self
    sizes = np.unique(px)
    votes = np.stack([(px[idx] == s).sum(1) for s in sizes], 1)
    px_pred = sizes[votes.argmax(1)]
    cls_pred = (cls[idx] == 2).mean(1) > 0.5
    rng = np.random.default_rng(0)
    sample = rng.choice(len(z), min(len(z), 4000), replace=False)
    return {
        "n_points": len(z), "n_sizes": len(sizes), "chance_px": round(1 / len(sizes), 3),
        "knn_px_acc": round(float((px_pred == px).mean()), 4),
        "knn_px_oct": round(float(np.abs(np.log2(px[idx] / px[:, None])).mean()), 4),
        "knn_same_obj": round(float((obj[idx] == obj[:, None]).mean()), 4),
        "knn_class_acc": round(float((cls_pred == (cls == 2)).mean()), 4),
        "sil_px": round(float(silhouette_score(z[sample], px[sample])), 4),
        "sil_class": round(float(silhouette_score(z[sample], cls[sample])), 4),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--feat-dir", type=Path, required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--stages", nargs="+", default=["enc5", "dec3"])
    ap.add_argument("--px-sizes", type=float, nargs="+", default=PX_DEFAULT, help="um/px")
    ap.add_argument("--band", type=float, nargs=2, default=[0.00236, 0.016],
                    help="um/px range for the second set of scores (default: multires band)")
    ap.add_argument("--objects-per-class", type=int, default=400, help="random objects per class")
    ap.add_argument("--pca", type=int, default=50)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--neighbors", type=int, default=30, help="UMAP n_neighbors")
    ap.add_argument("--min-dist", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    import umap  # umap-learn; imported here so --help works without it

    args.out_dir.mkdir(parents=True, exist_ok=True)
    coords_fh = open(args.out_dir / "umap_coords.csv", "w", newline="")
    cw = csv.DictWriter(coords_fh, fieldnames=["model", "stage", "object", "image", "subject", "class",
                                                "px_nm", "x", "y"])
    cw.writeheader()
    score_rows = []

    for m in args.models:
        F = Features(args.feat_dir / m, "obj")
        pis = [F.pi(p) for p in args.px_sizes]
        # Same object set for every stage and size: fully inside at every size, then a fixed sample.
        inside = (F.frac[pis] == 1).all(0)
        rng = np.random.default_rng(args.seed)
        objs = np.concatenate([
            rng.choice(np.flatnonzero(inside & (F.cls == c)),
                       min(args.objects_per_class, int((inside & (F.cls == c)).sum())), replace=False)
            for c in (1, 2)])
        objs.sort()
        print(f"{m}: {len(objs)} objects x {len(pis)} sizes", flush=True)

        for s in args.stages:
            f, _ = F.stage(s)
            x = np.concatenate([f[i, objs] for i in pis])                       # (sizes * objects, C)
            px = np.repeat(np.array(args.px_sizes) * 1000, len(objs)).round(2)
            obj = np.tile(objs, len(pis))
            cls = np.tile(F.cls[objs], len(pis))
            ok = ~np.isnan(x).any(1)
            x, px, obj, cls = x[ok], px[ok], obj[ok], cls[ok]

            z = PCA(n_components=min(args.pca, x.shape[1]), random_state=args.seed).fit_transform(
                StandardScaler().fit_transform(x))
            for subset, keep in (("all", np.ones(len(z), bool)),
                                 ("band", (px >= args.band[0] * 1000 - 1e-6) & (px <= args.band[1] * 1000 + 1e-6))):
                row = {"model": m, "stage": s, "subset": subset}
                row.update(scores(z[keep], px[keep], cls[keep], obj[keep], args.k))
                score_rows.append(row)
                print(f"  {s} {subset:4s} " + "  ".join(f"{k} {v}" for k, v in row.items()
                                                       if k.startswith(("knn", "sil"))), flush=True)

            emb = umap.UMAP(n_neighbors=args.neighbors, min_dist=args.min_dist,
                            random_state=args.seed).fit_transform(z)
            for j in range(len(emb)):
                cw.writerow({"model": m, "stage": s, "object": int(obj[j]), "image": int(F.image[obj[j]]),
                             "subject": F.subject[obj[j]], "class": int(cls[j]), "px_nm": px[j],
                             "x": round(float(emb[j, 0]), 4), "y": round(float(emb[j, 1]), 4)})

    coords_fh.close()
    with open(args.out_dir / "umap_scores.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(score_rows[0]))
        w.writeheader()
        w.writerows(score_rows)
    print(f"-> {args.out_dir}/umap_coords.csv, umap_scores.csv")


if __name__ == "__main__":
    main()
