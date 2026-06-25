# Resolution Invariance Experiment

Evaluates how well nnUNet-based axon/myelin segmentation models generalize across a wide range of pixel sizes (2.36 nm to 16 nm/px). Models are trained at one or multiple resolutions and tested on resampled versions of held-out images.

## Motivation

TEM images are acquired at varying resolutions across labs and datasets. A model trained at a single pixel size degrades at others. This experiment tests whether training on multiple resolutions (multires) solves this, and whether the DA5 data augmentation trainer adds anything on top.

---

## Models

| # | Name | Dataset | Trainer | Description |
|---|---|---|---|---|
| 1 | `witness` | TEM1 only | standard | Single-resolution baseline, trained at 4.9 nm/px |
| 2 | `multires` | TEM1 + TEM2 at 4 pixel sizes | standard | Multi-resolution training |
| 3 | `da5` | TEM1 only | DA5 | Single-resolution with DA5 augmentation trainer |
| 4 | `da5_multires` | TEM1 + TEM2 at 4 pixel sizes | DA5 | Multi-resolution with DA5 augmentation trainer |

DA5 trainer adds low-resolution simulation augmentation during training (`SimulateLowResolutionTransform` with zoom range 0.25–1.0) along with several intensity augmentations (brightness gradient, local gamma, sharpening, blank rectangles).

---

## Datasets

**TEM1**: Mouse optic nerve TEM images (NeuroPoly).
- Native pixel size: 2.36 nm/px
- 20 subjects, 4 held out for test (nyuMouse07, 11, 15, 25)
- GT: binary axon + myelin masks

**TEM2**: Mouse optic nerve TEM images (DANDI:001350).
- Native pixel size: 4.93 nm/px
- 86 images total; 10 with GT annotations across 5 subjects (sub-370, sub-372, sub-373C, sub-374, sub-375)
- Note: sub-373C chunks are small ROI crops with very few axons, Dice is unreliable for those

---

## Pixel sizes evaluated

16 log-spaced sizes from 2.36 nm to 16 nm/px:

```
0.00236, 0.0027058, 0.0032614, 0.003931, 0.004738, 0.00493,
0.0057108, 0.0068833, 0.0082966, 0.01, 0.0108148, 0.0116961,
0.0126491, 0.0136798, 0.0147945, 0.016
```

---

## Repository structure

```
training/
  train_witness.sh          # nnUNet training scripts
  train_multires.sh
  train_da5.sh
  train_da5_multires.sh
  prepare_dataset_witness.py   # dataset preparation
  prepare_dataset_multires.py
  evaluate_nnunet.py           # inference at multiple pixel sizes
  run_evaluation.sh            # TEM1 evaluation (models 1 & 2)
  run_evaluation_tem2.sh       # TEM2 evaluation (models 1 & 2)

evaluate_resinv.py          # ADS baseline evaluation (consistency metric)
recompute_metrics.py        # compute Dice vs GT from saved predictions
plot_resinv.py              # generate per-model and comparison plots
compute_morphometrics.py    # morphometrics (diameter, g-ratio) across pixel sizes
```

---

## Pipeline

### 1. Training

All models trained with nnUNet 2d, fold 0. Scripts in `training/`.

### 2. Evaluation

`evaluate_nnunet.py` takes a trained model and a data directory, resamples each image to all 16 pixel sizes, runs inference, and saves prediction masks.

```bash
python training/evaluate_nnunet.py \
    --model-dir <path/to/nnunet_model> \
    --model-name <name> \
    --data-dir <path/to/dataset> \
    --original-px 0.00493 \
    --output-dir <output_dir> \
    --gpu-id 0
```

Key flags:
- `--split-file`: JSON with `{"test_subjects": [...]}` to restrict evaluation to held-out subjects
- `--subjects`: filter to specific subjects without a split file
- `--gt-only`: when computing metrics, skip images with no GT mask
- `--models`: when recomputing metrics, restrict to specific model names

The script is resumable: images whose output PNGs already exist are skipped automatically.

### 3. Metrics

```bash
python recompute_metrics.py \
    --results-dir <output_dir> \
    --data-dir <dataset_root> \
    --models witness multires \
    --gt-only
```

Computes Dice against GT masks (axon, myelin) at every pixel size. GT masks expected at `{data_dir}/derivatives/labels/{subject}/micr/`.

### 4. Plotting

```bash
python plot_resinv.py --results-dir <output_dir> --exclude-labels uaxon
```

Generates per-model `results.png` and a `comparison.png` across all models.

### 5. Morphometrics

```bash
python compute_morphometrics.py
```

Computes mean axon diameter and g-ratio from prediction masks across all pixel sizes for a 10% sample of images. Outputs `morphometrics_tem1.csv/png` and `morphometrics_tem2.csv/png`.

