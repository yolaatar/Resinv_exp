# Release staging: multires_v2 (TEM1+TEM2, fold_all, nnUNet 2.8.1)

This is a staging area, not the final release repo. It packages the retrained
`multires_full_v2` checkpoint (Dataset006_TEM12_multires_v2, fold_all) into the
on-disk layout ADS expects for a downloadable model, following the same
structure as existing packaged models (e.g. `model_seg_unmyelinated_srf_app`
locally, or `model_seg_generalist` on GitHub): a flat nnU-Net trained-model
folder with `dataset.json` / `plans.json` at the root and one `fold_*/`
directory per fold, no `nnUNetTrainer__nnUNetPlans__2d/` wrapper.

## What's here

```
model_seg_generalist_tem_v2_light/
  dataset.json
  plans.json
  fold_all/
    checkpoint_final.pth
```

`model_seg_generalist_tem_v2_light` is a placeholder name, pick the real one
when this moves to its own repo (see Naming below).

## What's missing before this is publish-ready

- `dataset_fingerprint.json`, `debug.json`, `training_log_*.txt` — nnU-Net
  writes these alongside the checkpoint but they weren't pulled from tassan.
  Not required for inference (`nnUNetPredictor.initialize_from_trained_model_folder`
  only reads `dataset.json`, `plans.json`, and the checkpoint), but every other
  packaged ADS model ships them, so pull them from
  `~/resinv_exp/nnunet_resinv_v2/nnUNet_results/Dataset006_TEM12_multires_v2/nnUNetTrainer__nnUNetPlans__2d/fold_all/`
  on tassan for parity.
- No `_ensemble` variant — this is fold_all only (single fold), so only a
  `_light` package makes sense, same as `model_seg_generalist_light`.
- Not zipped or uploaded anywhere yet.

## Naming

ADS model folders follow `model_seg_<domain>_<task>_<modality>[_light|_ensemble]`
(see `AxonDeepSeg/model_cards.yaml` in the main repo). This model is a
retrain of the existing TEM `multires` model on nnUNet 2.8.1 instead of 2.2.1,
same data/task, not a new capability, so it's worth deciding with the team
whether it:
- replaces the current TEM generalist weights under the existing model card, or
- ships as a separate versioned entry (e.g. `model_seg_generalist_tem-v2`)
  so both are downloadable side by side for comparison.

## To actually publish (next repo)

1. Pull the missing files above, drop them into the matching folders here.
2. Rename the folder to the agreed `model_seg_..._light` name.
3. Zip it, create a GitHub release on the target model repo (new repo or an
   existing one), upload the zip as a release asset.
4. Add/update the entry in `AxonDeepSeg/model_cards.yaml`:

```yaml
<key>:
  full_name: <model_seg_...>
  task: myelinated-axon-segmentation
  n_classes: 2
  pixel_size: [0.00236, 0.00493]  # TEM1, TEM2 native — multires trains on both plus resampled copies
  model-info: TEM axon/myelin segmentation, multi-resolution training (TEM1+TEM2), nnUNet 2.8.1.
  training-data: TEM1 (NeuroPoly, 20 subjects) + TEM2 (DANDI:001350, all annotated subjects), trained at multiple pixel sizes for resolution invariance.
  weights:
    single_fold: <release URL>
    ensemble: ~
```

## Source

Trained via `training/train_multires_full_v2.sh` /
`training/prepare_dataset_multires_full_v2.py` in this repo. See
`run_commands.md` for the full training/eval command history and
`RESULTS.md` for Dice numbers vs. the v1 (nnUNet 2.2.1) checkpoint.
