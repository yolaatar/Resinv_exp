# ResInv — project context for Claude

## What this is
Resolution invariance experiment: evaluates how well nnUNet axon/myelin segmentation models generalize across pixel sizes (1 nm to 50 nm/px). Models are trained at one or multiple resolutions and tested on resampled images.

## Models
| Name | Dataset | Trainer | Description |
|---|---|---|---|
| `witness` | TEM1 only | standard | Single-res baseline, 4.9 nm/px, fold_0 |
| `multires` | TEM1 + TEM2 | standard | Multi-res training, fold_0 |
| `da5` | TEM1 only | DA5 | Single-res + DA5 augmentation, fold_0 |
| `da5_multires` | TEM1 + TEM2 | DA5 | Multi-res + DA5 augmentation, fold_0 |

## Pixel sizes evaluated
22 sizes from 1 nm to 50 nm/px (see `PX_SIZES` in `training/evaluate_nnunet.py`).

## Datasets
- **TEM1**: native 2.36 nm/px, 20 subjects, test split in `~/subject_split_tem1.json`
- **TEM2**: native 4.93 nm/px, 86 images total, only 10 have GT (sub-370, sub-372, sub-373C, sub-374, sub-375)

## Tassan paths (cluster)
- Scripts: `~/resinv_exp/scripts/`
- TEM1 data: `~/resinv_exp/data/TEM1`
- TEM2 data: `~/resinv_exp/data/TEM2`
- Models (witness, multires): `~/nnunet_results/Dataset00{1,2}_TEM_*/nnUNetTrainer__nnUNetPlans__2d/`
- DA5 models: `~/nnunet_da5_models/`
- TEM1 results: `~/resinv_exp/results_nnunet/`
- TEM2 results: `~/resinv_exp/results_nnunet_tem2/`
- venv: `~/resinv_exp/venv_resinv/`

## Running evaluation (tassan)
```bash
cd ~/resinv_exp/scripts && git pull
source ~/resinv_exp/venv_resinv/bin/activate

# TEM1 (witness + multires, test split)
bash ~/resinv_exp/scripts/training/run_evaluation.sh

# TEM2 (witness + multires, 10 GT images only)
bash ~/resinv_exp/scripts/training/run_evaluation_tem2.sh
```

Scripts are resumable — images with all predictions already saved are skipped.
Use tmux so runs survive SSH disconnects. Monitor with `tail -f ~/output_eval_*.log`.

## Important notes
- Always use **MONAI** for Dice metrics (`monai.metrics.compute_dice`), never a custom implementation.
- TEM2 evaluation uses `--gt-only` flag — only runs on images with GT masks in `derivatives/labels/`.
- Models use `fold_0` (not `fold_all`).
- See `run_commands.md` for full command history including training, metrics, and plotting.
