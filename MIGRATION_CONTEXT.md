# ResInv_exp — migration context (resolution invariance experiment)

Written 2026-08-15 to move active work to another machine. Read this top to bottom
before doing anything else on the new machine.

## 1. What this project is

Resolution invariance experiment for NeuroPoly (Julien Cohen-Adad's lab). Evaluates
how well nnUNet axon/myelin/unmyelinated-axon (uaxon) segmentation models generalize
across pixel sizes (1–50 nm/px), by resampling test images to 22 target pixel sizes
and running inference at each one.

Two parallel tracks live in this repo:
- **witness / multires / da5 / da5_multires** — original TEM1/TEM2 experiment,
  mostly finished earlier, documented in `CLAUDE.md` and `run_commands.md`.
- **armand_uaxon / armand_control** — this is the active track (all of this
  session's work). Evaluating Armand Collin's multires-trained 3-label model
  against his own publicly released "control" baseline, to show whether multires
  training genuinely buys resolution invariance. **This is what's summarized below.**

## 2. Repo locations

- **Local (this machine):** `C:\Users\Youssef\Documents\Cours\ADS\ResInv_exp`
- **GitHub remote:** `origin` → `https://github.com/yolaatar/Resinv_exp` (private,
  personal repo — NOT the public ADS org repo)
- **Current branch:** `main`, HEAD as of this writing: `2319e92`
  ("Add per-image prediction viz: control working-range sanity check and
  full-range multires vs control comparison")
- **Tassan (compute cluster), repo checkout:** `~/resinv_exp/scripts` on
  `tassan.neuro.polymtl.ca` — same GitHub repo, cloned there. Keep it in sync with
  `git pull origin main` before running anything (see §7 for a known CRLF gotcha).

### Known local git hygiene issue
`CLAUDE.md` and `README.md` show as modified in `git status` but `git diff` prints
nothing — this is a stale CRLF/LF flag, not real content drift (same signature as
the tassan issue in §7). Harmless, never resolved, don't worry about it.

Several small scripts existed only locally, uncommitted, until this migration prep
committed them (see §8 for the exact list). If you're on a fresh clone, `git pull`
gets you everything now.

## 3. Models under comparison

| Name | Model dir (tassan) | Dataset | Fold | Checkpoint | Labels |
|---|---|---|---|---|---|
| `armand_uaxon` | `~/nnunet_results/Dataset044_unmyelinated_armand_multires/nnUNetTrainer__nnUNetPlans__2d` | `Dataset044_unmyelinated_armand_multires` | `all` | `checkpoint_final.pth` | `{1: axon, 2: myelin, 3: uaxon}` |
| `armand_control` | `~/resinv_exp/models/control_v3/model_seg_unmyelinated_srf_app` | `Dataset043_TEM_UNMYELINATED_AGG` | `all` | `checkpoint_final.pth` | `{1: uaxon, 2: myelin, 3: axon, 4: nuclei(skip), 5: process(skip)}` |

- `armand_uaxon` = Armand Collin's multires-trained 3-label model (axon/myelin/uaxon),
  built for his own paper on axon/uaxon counting; the resolution-invariance angle is
  a bonus contribution I'm evaluating.
- `armand_control` = Armand's publicly released **v3.0.0** model
  (`model_seg_unmyelinated_srf_app`), used as the "no special resolution training"
  baseline.
  - **Verified this session (2026-08-15):** the exact model directory used for
    inference on tassan (`~/resinv_exp/models/control_v3/model_seg_unmyelinated_srf_app`)
    is byte-identical (`diff -rq` empty, SHA256 matched) to the official
    `v3.0.0` release asset at
    https://github.com/axondeepseg-pipeline/model_seg_unmyelinated_tem/releases/tag/v3.0.0
    — asset `model_seg_unmyelinated_srf_app.zip`, 345,135,031 bytes,
    SHA256 `f770b1895967080aec18e77b239596353e6f926757f6f704f4669e1558053306`,
    published 2026-06-26, commit `6dcf910`.

## 4. Test data

- Test set: `testset_TEM2` / "Armand's testset" — **21 images, 10 subjects**, BIDS layout.
  - Tassan: `~/resinv_exp/data/testset_armand_uaxon/`
  - Local: `data/testset_armand_uaxon/` (2.3 GB — NOT copied into this migration
    package, re-pull from tassan if needed, see §9)
- GT labels: `derivatives/labels/{subject}/micr/{sample}_seg-{axon|myelin|uaxon}-manual.png`
- Native/original pixel size of this dataset: **0.00493 μm/px (4.93 nm/px)** — this
  is also the training resolution for both models. Always passed as
  `--original-px 0.00493`.
- 22 target pixel sizes evaluated (μm/px), from `PX_SIZES` in `evaluate_nnunet.py`:
  ```
  0.001, 0.0015, 0.002,                                    # upsampling extrapolation
  0.00236, 0.0027058, 0.0032614, 0.003931, 0.004738, 0.00493,
  0.0057108, 0.0068833, 0.0082966, 0.01, 0.0108148, 0.0116961,
  0.0126491, 0.0136798, 0.0147945, 0.016,                  # native ADS resolution range
  0.020, 0.030, 0.050                                      # coarse extrapolation
  ```

## 5. Pipeline (4 stages, run on tassan)

All scripts live in `training/` in the repo (tassan: `~/resinv_exp/scripts/training/`,
EXCEPT the axon-count scripts which are flat under `~/resinv_exp/scripts/` on tassan,
not `~/resinv_exp/scripts/training/` — layout mismatch vs local repo, watch for it).

```bash
# on tassan
cd ~/resinv_exp/scripts && git pull origin main
source ~/resinv_exp/venv_resinv/bin/activate
export nnUNet_results=~/nnunet_results   # if not already set
```

### Stage 1 — inference (`evaluate_nnunet.py`)
```bash
bash ~/resinv_exp/scripts/training/run_evaluation_armand_uaxon.sh
bash ~/resinv_exp/scripts/training/run_evaluation_armand_control.sh
```
Writes `{img_name}_px{px}um_seg-{label}.png` per pixel size to
`~/resinv_exp/results_armand_uaxon/{model_name}/{img_name}/predictions/`.
Resumable (skips images with all pixel sizes already done) — but resume logic only
checks for `_seg-axon.png` existence, so after any labeling bug fix, wipe the whole
model's results dir before rerunning rather than trusting resume.

**Reads the label map dynamically from each model's own `dataset.json`** — see §7
for the bug this fixed.

### Stage 2 — Dice metrics (`recompute_metrics.py`, repo root)
```bash
cd ~/resinv_exp/scripts
python recompute_metrics.py \
  --results-dir ~/resinv_exp/results_armand_uaxon \
  --data-dir ~/resinv_exp/data/testset_armand_uaxon \
  --models armand_control \
  --gt-labels axon myelin uaxon \
  --workers 32 \
  2>&1 | tee ~/output_dice_armand_control.log
```
(swap `--models armand_uaxon` for the other model). Writes
`results_armand_uaxon/{model}/results.csv`. Always uses MONAI
(`monai.metrics.compute_dice`) for Dice, never a custom implementation.

### Stage 3 — morphometrics (`training/run_morphometrics.py`)
```bash
python training/run_morphometrics.py \
  --results-dir ~/resinv_exp/results_armand_uaxon/armand_control \
  --data-dir ~/resinv_exp/data/testset_armand_uaxon \
  --original-px 0.00493 \
  --model-name armand_control
```
Regenerates resampled raw images + combined axonmyelin masks, shells out to the
real `axondeepseg_morphometrics` CLI once per pixel size (myelinated + `-u`
unmyelinated pass), aggregates into `morphometrics_summary.csv`.

### Stage 4 — axon/uaxon counts
```bash
bash ~/resinv_exp/scripts/run_axon_counts.sh \
  ~/resinv_exp/results_armand_uaxon/armand_control \
  ~/resinv_exp/scripts
```
Note: scripts dir here is `~/resinv_exp/scripts` (flat), not `.../scripts/training`.
Writes `axon_counts_all.csv`.

## 6. Pulling results locally

```bash
scp -r yolaa@tassan.neuro.polymtl.ca:~/resinv_exp/results_armand_uaxon/armand_control/results.csv <local>
scp -r yolaa@tassan.neuro.polymtl.ca:~/resinv_exp/results_armand_uaxon/armand_control/morphometrics_summary.csv <local>
scp -r yolaa@tassan.neuro.polymtl.ca:~/resinv_exp/results_armand_uaxon/armand_control/axon_counts_all.csv <local>
```
**Gotcha:** `scp -r` on the whole model directory has silently dropped `results.csv`
and `axon_counts_all.csv` before (only per-image prediction folders + one CSV came
through). Always verify pulled file list against `ls -la` on tassan rather than
trusting a recursive pull completed.

## 7. Bugs found and fixed this session

### 7a. Hardcoded label map in `evaluate_nnunet.py` (root cause, fixed, committed `5f8b505`)
Before: module-level `LABEL_MAP = {1: "axon", 2: "myelin"}`, used for every model
regardless of its actual `dataset.json` label scheme, and `dataset.json` was never
read at all. This silently dropped/mislabeled the uaxon class for `armand_control`
(whose scheme is `{background:0, uaxon:1, myelin:2, axon:3, nuclei:4, process:5}`),
so the first inference run produced 0 uaxon prediction files across all 21 images ×
22 pixel sizes.

Fix: `load_label_map(model_dir)` builds `{class_id: label_name}` from the model's own
`dataset.json["labels"]`, with `DEFAULT_LABEL_MAP = {1: "axon", 2: "myelin"}` as a
fallback only if `dataset.json` is missing, and `LABEL_SKIP = {"nuclei", "process"}`
to exclude labels not consumed downstream. Verified via log line
`Label map : {1: 'uaxon', 2: 'myelin', 3: 'axon'}` and 462/462 (21 images × 22 px)
uaxon files written after a clean rerun.

### 7b. CRLF/LF false-diff pattern (recognized, not a bug — watch for this again)
A `git diff` showing an entire file as fully removed+added, but where every `-` line
has an identical-content `+` counterpart, is a line-ending mismatch (Windows `scp`
origin touching a file on the Linux tassan checkout), not real corruption. Fix when
it blocks a `git pull`: `git checkout -- <file> && git pull origin main`.

### 7c. Tassan script layout mismatch
Axon-count helper scripts (`run_axon_counts.sh`, `count_axons.py`,
`aggregate_axon_counts.py`) live flat under `~/resinv_exp/scripts/` on tassan, NOT
under `~/resinv_exp/scripts/training/` like the local repo structure suggests.

## 8. Files committed to git this session (all in `origin/main` now)

- `training/evaluate_nnunet.py` — label-map fix (`5f8b505`)
- `tmp_viz/plot_comparison.py` — multires-vs-control comparison plots (`582a01d`)
- `tmp_viz/viz_control_working_range.py`,
  `tmp_viz/viz_model_comparison_full_range.py` — per-image prediction viz (`2319e92`)
- **Just committed as part of this migration prep** (were local-only before,
  now pushed): `training/run_evaluation_armand_control.sh`,
  `training/run_evaluation_armand_uaxon.sh`, `training/run_morphometrics.py`,
  `training/run_axon_counts.sh`, `training/aggregate_axon_counts.py`,
  `tmp_viz/make_full_range_comparison.py`, `tmp_viz/make_viz.py` (older
  TEM1/TEM2 witness-vs-multires viz scripts, unrelated to armand track but
  previously undocumented in git).

## 9. What's NOT in git / needs re-fetching on the new machine

Too large for git, need to `scp` from tassan again if the new machine needs them:
- `data/` (2.3 GB) — raw test images, BIDS structure. Source of truth: tassan
  `~/resinv_exp/data/`.
- `nnunet_results/`, `nnunet_results_v2/` (355 MB + 708 MB) — model checkpoints.
  Source of truth: tassan `~/nnunet_results/`.
- `results_armand_uaxon/armand_control/` per-image prediction folders (278 MB) —
  full predictions for all 21 images at 22 pixel sizes were pulled locally this
  session (used for the viz scripts). Source of truth: tassan
  `~/resinv_exp/results_armand_uaxon/armand_control/`.
- `results_armand_uaxon/armand_uaxon/` per-image predictions — **only 1 of 21
  images** (`sub-366A_sample-0001_acq-roi_TEM`) was pulled locally (16 MB), just
  enough for the comparison viz. The other 20 images' predictions exist on tassan
  (`~/resinv_exp/results_armand_uaxon/armand_uaxon/`) but were never pulled.
- `tmp_viz/sub-372_sample-0002_acq-roi_TEM/` (13 MB) — leftover raw image data
  from earlier ad-hoc viz work, not committed, not essential.

## 10. What IS small enough to hand-carry (the upload package)

Everything under `results_armand_uaxon/` **except** the two big per-image
prediction folders — i.e. all CSVs and all PNG plots, ~10 MB total:

```
results_armand_uaxon/
  armand_uaxon/results.csv
  armand_uaxon/morphometrics_summary.csv
  armand_uaxon/axon_counts_all.csv
  armand_uaxon/dice_vs_pixelsize.png
  armand_uaxon/morphometrics_vs_pixelsize.png
  armand_uaxon/axon_counts_vs_pixelsize.png
  armand_control/results.csv
  armand_control/morphometrics_summary.csv
  armand_control/axon_counts_all.csv
  dice_comparison.png
  gratio_comparison.png
  counts_comparison.png
  control_working_range_predictions.png
  model_comparison_full_range.png
```
These are copied into `migration_package/results_armand_uaxon/` alongside this
file — zip that whole folder and upload it, or just this MD file if you're fine
regenerating plots from the CSVs (all plotting scripts are in git under
`tmp_viz/`).

## 11. Results so far (headline finding)

`armand_control` (v3.0.0 baseline, no explicit multi-resolution training) performs
**as well as** `armand_uaxon` (multires-trained) in the 2–8 nm/px band around the
shared 4.93 nm/px training resolution — e.g. axon Dice at 3.9 nm/px: 0.975 (control)
vs 0.971 (multires). The gap opens sharply **outside** that band:
- Below 2 nm/px: axon Dice 0.484 (control) vs 0.963 (multires) at 1 nm/px.
- Above ~10 nm/px: axon Dice 0.329 (control) vs 0.647 (multires) at 20 nm/px;
  control's g-ratio also degrades much faster (myelin under-detection) and its
  axon count over-segments sharply at the fine end (>90 near 1–2 nm/px).

This was cross-checked two ways this session, both hold up:
- **Numerically**: recomputed both models' `results.csv` directly, confirmed
  identical image sets / pixel-size sets / NaN patterns (no pipeline asymmetry).
- **Visually**: `model_comparison_full_range.png` shows the same pattern on real
  prediction overlays — clean/near-identical segmentation in the 2–8 nm band for
  both models, control losing myelin and over-classifying uaxon at the coarse end,
  noisier control segmentation at the fine end.

**Working hypothesis** (not yet confirmed against the control model's actual
training config): standard nnUNet default scale augmentation gives `armand_control`
some resolution robustness within roughly ±2x of its training pixel size, while
`armand_uaxon`'s explicit multires training extends that much further in both
directions. To confirm, would need to check `armand_control`'s `plans.json` /
training log on tassan for its augmentation setup — not yet done.

## 12. Open threads / not yet done

- Confirming the augmentation-config hypothesis above (§11) by inspecting
  `armand_control`'s `plans.json` on tassan.
- Never explicitly re-requested, but flagged earlier in the broader session as
  eventual follow-ups (don't act on these without the user asking again):
  - Migrating evaluation scripts/documentation into Armand's own repo
    (`model_seg_unmyelinated_tem`) once results look good.
  - Reviewing other commits that landed via earlier `git pull` on tassan
    (multires_tem12 model, TEM1+TEM2 joint training, GPU-parallelized eval
    scripts) — never actually reviewed.

## 13. Cluster access reminder

No direct SSH/scp access from Claude in this environment — every tassan command in
this doc needs to be run by the user and pasted back. `ssh yolaa@tassan.neuro.polymtl.ca`,
venv at `~/resinv_exp/venv_resinv/`.
