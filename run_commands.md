# Run commands

## Transfer scripts to tassan (on your Mac)

```bash
rsync -avz /Users/yolaatar/Developer/ADS/resinv/training/ yolaa@tassan.neuro.polymtl.ca:~/resinv_exp/scripts/training/
```

---

## nnUNet 2.2.1 + PyTorch 2.10 patches (one-time, on tassan)

**Patch 1 — polylr verbose arg (training):**
```bash
sed -i 's/super().__init__(optimizer, current_step if current_step is not None else -1, False)/super().__init__(optimizer, current_step if current_step is not None else -1)/' ~/resinv_exp/venv_resinv/lib/python3.12/site-packages/nnunetv2/training/lr_scheduler/polylr.py
```

**Patch 2 — weights_only in trainer (training resume):**
```bash
sed -i "s/torch.load(filename_or_checkpoint, map_location=self.device)/torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)/" ~/resinv_exp/venv_resinv/lib/python3.12/site-packages/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py
```

**Patch 3 — weights_only in predictor (inference):**
```bash
sed -i "s/torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),/torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name), weights_only=False,/" ~/resinv_exp/venv_resinv/lib/python3.12/site-packages/nnunetv2/inference/predict_from_raw_data.py
```

---

## Training (on tassan)

Activate venv first: `source ~/resinv_exp/venv_resinv/bin/activate`

| Model | Script | GPU | Log |
|-------|--------|-----|-----|
| 1 — Witness | `bash ~/resinv_exp/scripts/training/train_witness.sh` | 0 | `~/output_witness.log` |
| 2 — Multi-res | `bash ~/resinv_exp/scripts/training/train_multires.sh` | 1 | `~/output_multires.log` |
| 3 — DA5 | `bash ~/resinv_exp/scripts/training/train_da5.sh` | 0 | `~/output_da5.log` |
| 4 — DA5 + Multi-res | `bash ~/resinv_exp/scripts/training/train_da5_multires.sh` | 1 | `~/output_da5_multires.log` |

Checkpoints: `~/duke/temp/yolaatar/nnunet_resinv/nnUNet_results/`

---

## Evaluation — nnUNet models on TEM1 test set (on tassan)

### Models 1 and 2

```bash
source ~/resinv_exp/venv_resinv/bin/activate
bash ~/resinv_exp/scripts/training/run_evaluation.sh
```

### Models 3 and 4

```bash
source ~/resinv_exp/venv_resinv/bin/activate

CUDA_VISIBLE_DEVICES=1 python ~/resinv_exp/scripts/training/evaluate_nnunet.py \
    --model-dir ~/duke/temp/yolaatar/nnunet_resinv/nnUNet_results/Dataset001_TEM_witness/nnUNetTrainerDA5__nnUNetPlans__2d \
    --model-name da5 \
    --data-dir ~/duke/temp/yolaatar/resinv_exp/data/TEM1 \
    --output-dir ~/duke/temp/yolaatar/resinv_exp/results_nnunet \
    --gpu-id 0 \
    2>&1 | tee ~/output_eval_da5.log

CUDA_VISIBLE_DEVICES=1 python ~/resinv_exp/scripts/training/evaluate_nnunet.py \
    --model-dir ~/duke/temp/yolaatar/nnunet_resinv/nnUNet_results/Dataset002_TEM_multires/nnUNetTrainerDA5__nnUNetPlans__2d \
    --model-name da5_multires \
    --data-dir ~/duke/temp/yolaatar/resinv_exp/data/TEM1 \
    --output-dir ~/duke/temp/yolaatar/resinv_exp/results_nnunet \
    --gpu-id 0 \
    2>&1 | tee ~/output_eval_da5_multires.log
```

### TEM2 evaluation (all 86 images, 0.00493 μm/px native, upsampling for finer resolutions)

```bash
source ~/resinv_exp/venv_resinv/bin/activate

CUDA_VISIBLE_DEVICES=1 python ~/resinv_exp/scripts/training/evaluate_nnunet.py \
    --model-dir ~/duke/temp/yolaatar/nnunet_resinv/nnUNet_results/Dataset001_TEM_witness/nnUNetTrainer__nnUNetPlans__2d \
    --model-name witness \
    --data-dir ~/duke/temp/yolaatar/resinv_exp/data/TEM2 \
    --original-px 0.00493 \
    --output-dir ~/duke/temp/yolaatar/resinv_exp/results_nnunet_tem2 \
    --gpu-id 0 \
    2>&1 | tee ~/output_eval_tem2_witness.log

CUDA_VISIBLE_DEVICES=1 python ~/resinv_exp/scripts/training/evaluate_nnunet.py \
    --model-dir ~/duke/temp/yolaatar/nnunet_resinv/nnUNet_results/Dataset002_TEM_multires/nnUNetTrainer__nnUNetPlans__2d \
    --model-name multires \
    --data-dir ~/duke/temp/yolaatar/resinv_exp/data/TEM2 \
    --original-px 0.00493 \
    --output-dir ~/duke/temp/yolaatar/resinv_exp/results_nnunet_tem2 \
    --gpu-id 0 \
    2>&1 | tee ~/output_eval_tem2_multires.log
```

---

## Retrieve and process results (on your Mac)

### Retrieve nnUNet results

```bash
rsync -avz yolaa@tassan.neuro.polymtl.ca:~/duke/temp/yolaatar/resinv_exp/results_nnunet/ /Users/yolaatar/Developer/ADS/resinv/results_nnunet/
```

### Retrieve nnUNet TEM2 results

```bash
rsync -avz yolaa@tassan.neuro.polymtl.ca:~/duke/temp/yolaatar/resinv_exp/results_nnunet_tem2/ /Users/yolaatar/Developer/ADS/resinv/results_nnunet_tem2/
```

### Retrieve ADS baseline results (TEM1)

```bash
rsync -avz yolaa@tassan.neuro.polymtl.ca:~/duke/temp/yolaatar/resinv_exp/results_tem1/ /Users/yolaatar/Developer/ADS/resinv/results_tem1/
```

### Recompute metrics

```bash
cd /Users/yolaatar/Developer/ADS/resinv
source ../axondeepseg/.venv/bin/activate

# nnUNet models
python recompute_metrics.py --results-dir ./results_nnunet --data-dir /Users/yolaatar/Developer/ADS/data/TEM1

# ADS baseline
python recompute_metrics.py --results-dir ./results_tem1 --data-dir /Users/yolaatar/Developer/ADS/data/TEM1
```

### Plot

```bash
cd /Users/yolaatar/Developer/ADS/resinv
source ../axondeepseg/.venv/bin/activate
python plot_resinv.py --results-dir ./results_nnunet
```
