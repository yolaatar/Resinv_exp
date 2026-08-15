# Release staging: model_seg_tem-multires (TEM1+TEM2, fold_all, nnUNet 2.8.1)

This is a staging area, not the final release repo. It packages the retrained
`multires_full_v2` checkpoint (Dataset006_TEM12_multires_v2, fold_all) into the
on-disk layout ADS expects for a downloadable model, following the same
structure as existing packaged models (e.g. `model_seg_unmyelinated_srf_app`
locally, or `model_seg_generalist` on GitHub): a flat nnU-Net trained-model
folder with `dataset.json` / `plans.json` at the root and one `fold_*/`
directory per fold, no `nnUNetTrainer__nnUNetPlans__2d/` wrapper.

## What's here

```
model_seg_tem-multires_light/
  dataset.json
  plans.json
  fold_all/
    checkpoint_final.pth
    dataset_fingerprint.json
    debug.json
    training_log_2026_7_30_20_06_27.txt
```

`dataset_fingerprint.json`, `debug.json`, and the training log were pulled
from tassan (`~/resinv_exp/nnunet_resinv_v2/nnUNet_results/Dataset006_TEM12_multires_v2/`,
note `dataset_fingerprint.json` lives at the dataset root, the other two are
under `nnUNetTrainer__nnUNetPlans__2d/fold_all/`) for parity with other
packaged ADS models. Not required for inference
(`nnUNetPredictor.initialize_from_trained_model_folder` only reads
`dataset.json`, `plans.json`, and the checkpoint).

## What's missing before this is publish-ready

- No `_ensemble` variant — this is fold_all only (single fold), so only a
  `_light` package makes sense, same as `model_seg_generalist_light`.
- Add the `model_cards.yaml` entry below to `AxonDeepSeg/model_cards.yaml`.

## Published

- Repo: https://github.com/axondeepseg/model_seg_tem-multires
- Release: https://github.com/axondeepseg/model_seg_tem-multires/releases/tag/v1.0
- Asset: https://github.com/axondeepseg/model_seg_tem-multires/releases/download/v1.0/model_seg_tem-multires_light.zip

## Naming

Ships as a new, separate model (not replacing the existing TEM generalist
weights) — `model_seg_tem-multires`, package name `model_seg_tem-multires_light`.
Drops the task (`axon-myelin`) from the name since `generalist` doesn't spell
it out either; keeps `TEM` (modality) and `multires` (the property that
actually distinguishes this checkpoint: trained across multiple pixel sizes,
robust to resolution).

## To actually publish

1. ~~Pull the missing files, drop them into the matching folders here.~~ Done.
2. ~~Zip `model_seg_tem-multires_light/`, create a GitHub release, upload the
   zip as a release asset.~~ Done, see Published above.
3. Add this entry to `AxonDeepSeg/model_cards.yaml` (not yet done):

```yaml
tem-multires:
  full_name: model_seg_tem-multires
  task: myelinated-axon-segmentation
  n_classes: 2
  pixel_size: [0.00236, 0.00493]  # TEM1, TEM2 native — multires trains on both plus resampled copies
  model-info: TEM axon/myelin segmentation, multi-resolution training (TEM1+TEM2), nnUNet 2.8.1.
  training-data: TEM1 (NeuroPoly, 20 subjects) + TEM2 (DANDI:001350, all annotated subjects), trained at multiple pixel sizes for resolution invariance.
  weights:
    single_fold: https://github.com/axondeepseg/model_seg_tem-multires/releases/download/v1.0/model_seg_tem-multires_light.zip
    ensemble: ~
```

## Source

Trained via `training/train_multires_full_v2.sh` /
`training/prepare_dataset_multires_full_v2.py` in this repo. See
`run_commands.md` for the full training/eval command history and
`RESULTS.md` for Dice numbers vs. the v1 (nnUNet 2.2.1) checkpoint.
