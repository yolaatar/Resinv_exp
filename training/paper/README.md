# Paper retrain batch (nnUNet 2.8.1)

Clean retrain of every model the paper compares, with one protocol so differences come from
the training data or trainer, not from how each model happened to be set up.

## What's held fixed

| Thing | How |
|---|---|
| nnUNet version | `train_paper.sh` refuses to run unless the venv has nnunetv2 2.8.1 |
| Network, patch, batch | One set of plans for everyone, planned on `Dataset102_TEM1_multires4` and transferred with `nnUNetv2_move_plans_between_datasets`. Planned separately, nnUNet would size each network from its own images, so witness and multires would get different architectures |
| Test subjects | TEM1 `subject_split.json` (seed 42) is reused, never regenerated |
| Validation images | `generate_splits_shared.py`: fold k validates on the same source images in every dataset (the old GroupKFold splits don't guarantee this) |
| Dataset construction | `prepare_dataset_paper.py` builds every variant, same interpolation as v2 |
| Training budget | stock 1000 epochs x 250 iterations; evaluation always uses `checkpoint_final.pth` |

## Models

`bash train_paper.sh --list` prints the registry. Summary:

| Batch | Key | Data | Trainer | Question |
|---|---|---|---|---|
| pilot | witness f0 | TEM1 2.36 nm | stock | Fair pair with the existing multires_v2, fast |
| 1 | witness, multires | TEM1 2.36 / + 7, 10, 16 nm | stock | Core comparison, folds 0-2 for error bars |
| 1 | da5, da5_multires | same | nnUNetTrainerDA5 | Blur augmentation vs scale |
| 2 | scaleaug | TEM1 2.36 nm | ScaleAug2p5 (online zoom to 1/2.5) | Does the robust band follow the scales seen in training? |
| 2 | multires2 / multires8 | + 16 nm / + 7 log-spaced sizes to 16 nm | stock | Same coverage, sparser / denser |
| 2 | multires_to7 | + 3.4, 4.9, 7 nm | stock | Same count as multires4, half the coverage |
| 3 | witness_resenc, multires_resenc | batch 1 data | stock, ResEnc M plans | Does the effect hold for a residual encoder? |
| gated | witness_tem2, multires_tem2 | TEM2 4.93 nm | stock | Reverse direction. Needs a leakage-free TEM2 split first |

## Running on tassan

```bash
cd ~/resinv_exp/scripts && git fetch origin && git checkout paper-training-scripts   # or main once merged
source ~/resinv_exp/venv_resinv_v2/bin/activate
cd training/paper

# pilot: one run, gives a fair 2.8.1 witness to compare against multires_v2 right away
bash launch_batch.sh pilot 0

# batch 1 over both GPUs (8 runs, ~6 h each, ~24 h wall)
bash launch_batch.sh 1
# or everything on GPU 1 only
bash launch_batch.sh 1 1

tmux ls                          # sessions are paper_b<batch>_gpu<N>
tail -f ~/resinv_exp/nnunet_paper/logs/queue_gpu0.log
```

Batch 2 installs the custom trainers automatically (`install_custom_trainers.sh`). Batch 1
doesn't need them.

Evaluate a finished model on both test sets at all 22 pixel sizes:

```bash
bash run_eval_paper.sh witness 0 0
# -> ~/resinv_exp/results_paper/{tem1,tem2test}/witness_f0/results.csv
```

## Where things go

- raw datasets, results: `~/resinv_exp/nnunet_paper/{nnUNet_raw,nnUNet_results}`
- preprocessed: `/tmp/yolaatar/nnunet_preprocessed_paper` (same convention as v2; override with
  `RESINV_PAPER_PREPROCESSED`). `/tmp` may be wiped on reboot, in which case the next run
  re-preprocesses automatically.
- logs: `~/resinv_exp/nnunet_paper/logs/`
- cost log: `~/resinv_exp/nnunet_paper/cost_log.jsonl`, one line per finished run with
  preprocessing seconds, training seconds, case count and preprocessed disk size

## Things worth knowing

- **Multires is 4x the cases but only ~1.19x the pixels.** The 7, 10 and 16 nm copies are
  0.11x, 0.06x and 0.02x the native pixel count. Training time doesn't grow with dataset size
  either (fixed iteration count), so the v2 logs show multires_full_v2 (632 cases) and
  multires_v2 (512) both taking ~5.7 h.
- **The 10 and 16 nm copies are smaller than the 768x1152 patch** (888x539 and 555x337), so
  each is a whole image inside one padded patch. nnUNet samples cases uniformly, so about half
  of multires4's training patches are entire padded low-res images.
- **Online zoom-out is capped by image size.** TEM1 images are 3762x2286; zooming out by s needs
  an s x patch crop, so s past ~2.9 samples padding. That's why ScaleAug stops at 2.5 and why
  the full 2.36 to 16 nm range (s ~ 6.8) needs offline copies.
- nnUNet treats every PNG as spacing 1, so it never resamples inputs internally. The only
  resolution handling is what the training data and augmentation provide.
