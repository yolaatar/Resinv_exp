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

### Model 4 resume — write to local home to avoid duke I/O crashes

```bash
mkdir -p ~/nnunet_results_tmp/Dataset002_TEM_multires/nnUNetTrainerDA5__nnUNetPlans__2d/fold_0
cp ~/duke/temp/yolaatar/nnunet_resinv/nnUNet_results/Dataset002_TEM_multires/nnUNetTrainerDA5__nnUNetPlans__2d/fold_0/checkpoint_best.pth ~/nnunet_results_tmp/Dataset002_TEM_multires/nnUNetTrainerDA5__nnUNetPlans__2d/fold_0/checkpoint_best.pth
cp ~/duke/temp/yolaatar/nnunet_resinv/nnUNet_results/Dataset002_TEM_multires/nnUNetTrainerDA5__nnUNetPlans__2d/fold_0/checkpoint_latest.pth ~/nnunet_results_tmp/Dataset002_TEM_multires/nnUNetTrainerDA5__nnUNetPlans__2d/fold_0/checkpoint_latest.pth
```

Resume:
```bash
export nnUNet_raw="${HOME}/duke/temp/yolaatar/nnunet_resinv/nnUNet_raw"
export nnUNet_preprocessed="/tmp/yolaatar/nnunet_preprocessed"
export nnUNet_results="${HOME}/nnunet_results_tmp"
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 2 2d 0 -tr nnUNetTrainerDA5 --c 2>&1 | tee -a ~/output_da5_multires.log
```

Sync to duke when done:
```bash
rsync -avz ~/nnunet_results_tmp/ ~/duke/temp/yolaatar/nnunet_resinv/nnUNet_results/
```

---

## Evaluation — models 1 & 2, extended pixel sizes (1–50 nm), on tassan

Scripts are resumable — already-done images are skipped automatically.
Models: `~/nnunet_results/Dataset00{1,2}_TEM_*/`
TEM1 data: `~/resinv_exp/data/TEM1` + split file `~/subject_split_tem1.json` (test split only)
TEM2 data: `~/resinv_exp/data/TEM2` (GT subjects only: sub-370, sub-372, sub-373C, sub-374, sub-375)

### Step 1 — pull latest scripts

```bash
cd ~/resinv_exp/scripts && git pull
```

### Step 2 — TEM1 (native 2.36 nm/px, test split)

```bash
tmux new -s eval_tem1
source ~/resinv_exp/venv_resinv/bin/activate
bash ~/resinv_exp/scripts/training/run_evaluation.sh
```

Logs: `~/output_eval_witness.log`, `~/output_eval_multires.log`

### Step 3 — TEM2 (native 4.93 nm/px, 10 GT images across 5 subjects)

```bash
tmux new -s eval_tem2
source ~/resinv_exp/venv_resinv/bin/activate
bash ~/resinv_exp/scripts/training/run_evaluation_tem2.sh
```

Logs: `~/output_eval_tem2_witness.log`, `~/output_eval_tem2_multires.log`

---

## Evaluation — nnUNet models on TEM1 test set (on tassan)

### Models 1 and 2

```bash
source ~/resinv_exp/venv_resinv/bin/activate
bash ~/resinv_exp/scripts/training/run_evaluation.sh
```

### Models 3 and 4

Run in parallel on 2 GPUs. First sync model 4 from local home:

```bash
rsync -avz ~/nnunet_results_tmp/ ~/duke/temp/yolaatar/nnunet_resinv/nnUNet_results/
```

Note: DA5 model dirs must be copied from joplin first (duke permission issue on tassan):
```bash
# On joplin
rsync -avz --mkpath ~/duke/temp/yolaatar/nnunet_resinv/nnUNet_results/Dataset001_TEM_witness/nnUNetTrainerDA5__nnUNetPlans__2d/ yolaa@ge.polymtl.ca@tassan:~/nnunet_da5_models/Dataset001_TEM_witness/nnUNetTrainerDA5__nnUNetPlans__2d/
rsync -avz --mkpath ~/duke/temp/yolaatar/nnunet_resinv/nnUNet_results/Dataset002_TEM_multires/nnUNetTrainerDA5__nnUNetPlans__2d/ yolaa@ge.polymtl.ca@tassan:~/nnunet_da5_models/Dataset002_TEM_multires/nnUNetTrainerDA5__nnUNetPlans__2d/
```

**GPU 0 — model 3 (DA5), TEM1 then TEM2:**
```bash
tmux new -s eval_da5
source ~/resinv_exp/venv_resinv/bin/activate

CUDA_VISIBLE_DEVICES=0 python ~/resinv_exp/scripts/training/evaluate_nnunet.py \
    --model-dir ~/nnunet_da5_models/Dataset001_TEM_witness/nnUNetTrainerDA5__nnUNetPlans__2d \
    --model-name da5 \
    --data-dir ~/duke/temp/yolaatar/resinv_exp/data/TEM1 \
    --output-dir ~/duke/temp/yolaatar/resinv_exp/results_nnunet \
    --max-images 40 --gpu-id 0 2>&1 | tee ~/output_eval_da5.log && \
CUDA_VISIBLE_DEVICES=0 python ~/resinv_exp/scripts/training/evaluate_nnunet.py \
    --model-dir ~/nnunet_da5_models/Dataset001_TEM_witness/nnUNetTrainerDA5__nnUNetPlans__2d \
    --model-name da5 \
    --data-dir ~/duke/temp/yolaatar/resinv_exp/data/TEM2/001350 \
    --original-px 0.00493 \
    --subjects sub-370 sub-372 sub-373C sub-374 sub-375 \
    --output-dir ~/duke/temp/yolaatar/resinv_exp/results_nnunet_tem2 \
    --max-images 40 --gpu-id 0 2>&1 | tee ~/output_eval_da5_tem2.log
```

**GPU 1 — model 4 (DA5+multires), TEM1 then TEM2:**

If duke top-level listing fails on tassan, copy split file from joplin first:
```bash
# On joplin
scp ~/duke/temp/yolaatar/resinv_exp/data/TEM1/subject_split.json yolaa@ge.polymtl.ca@tassan:~/subject_split_tem1.json
```

```bash
tmux new -s eval_da5_multires
source ~/resinv_exp/venv_resinv/bin/activate
cd ~/resinv_exp/scripts/training && git pull

CUDA_VISIBLE_DEVICES=1 python ~/resinv_exp/scripts/training/evaluate_nnunet.py \
    --model-dir ~/nnunet_da5_models/Dataset002_TEM_multires/nnUNetTrainerDA5__nnUNetPlans__2d \
    --model-name da5_multires \
    --data-dir ~/duke/temp/yolaatar/resinv_exp/data/TEM1 \
    --split-file ~/subject_split_tem1.json \
    --output-dir ~/duke/temp/yolaatar/resinv_exp/results_nnunet \
    --max-images 40 --gpu-id 0 2>&1 | tee ~/output_eval_da5_multires_tem1.log && \
CUDA_VISIBLE_DEVICES=1 python ~/resinv_exp/scripts/training/evaluate_nnunet.py \
    --model-dir ~/nnunet_da5_models/Dataset002_TEM_multires/nnUNetTrainerDA5__nnUNetPlans__2d \
    --model-name da5_multires \
    --data-dir ~/duke/temp/yolaatar/resinv_exp/data/TEM2/001350 \
    --original-px 0.00493 \
    --subjects sub-370 sub-372 sub-373C sub-374 sub-375 \
    --output-dir ~/duke/temp/yolaatar/resinv_exp/results_nnunet_tem2 \
    --max-images 40 --gpu-id 0 2>&1 | tee ~/output_eval_da5_multires_tem2.log

### Model 4 TEM2 — resume on GPU 0 (with skip logic)

Duke listing fails from tmux on tassan — copy GT subjects to local first (from normal SSH session):
```bash
mkdir -p ~/tem2_gt_data
cp -r ~/duke/temp/yolaatar/resinv_exp/data/TEM2/001350/sub-370 ~/duke/temp/yolaatar/resinv_exp/data/TEM2/001350/sub-372 ~/duke/temp/yolaatar/resinv_exp/data/TEM2/001350/sub-373C ~/duke/temp/yolaatar/resinv_exp/data/TEM2/001350/sub-374 ~/duke/temp/yolaatar/resinv_exp/data/TEM2/001350/sub-375 ~/tem2_gt_data/
```

Then from tmux:
```bash
CUDA_VISIBLE_DEVICES=0 python ~/resinv_exp/scripts/training/evaluate_nnunet.py \
    --model-dir ~/nnunet_da5_models/Dataset002_TEM_multires/nnUNetTrainerDA5__nnUNetPlans__2d \
    --model-name da5_multires \
    --data-dir ~/tem2_gt_data \
    --original-px 0.00493 \
    --output-dir ~/duke/temp/yolaatar/resinv_exp/results_nnunet_tem2 \
    --gpu-id 0 2>&1 | tee -a ~/output_eval_da5_multires_tem2.log
```

Cleanup after:
```bash
rm -rf ~/tem2_gt_data
```
```

### Download TEM2 from DANDI (on tassan, one-time)

```bash
pip install dandi -q
mkdir -p ~/duke/temp/yolaatar/resinv_exp/data/TEM2
dandi download --output-dir ~/duke/temp/yolaatar/resinv_exp/data/TEM2 DANDI:001350/0.250511.1527
```

### TEM2 evaluation — models 1 and 2 (86 images, 0.00493 μm/px native)

```bash
source ~/resinv_exp/venv_resinv/bin/activate
bash ~/resinv_exp/scripts/training/run_evaluation_tem2.sh
```

Logs: `~/output_eval_tem2_witness.log`, `~/output_eval_tem2_multires.log`

### TEM2 — multires only (run separately if needed)

```bash
CUDA_VISIBLE_DEVICES=0 python ~/resinv_exp/scripts/training/evaluate_nnunet.py \
    --model-dir ~/duke/temp/yolaatar/nnunet_resinv/nnUNet_results/Dataset002_TEM_multires/nnUNetTrainer__nnUNetPlans__2d \
    --model-name multires \
    --data-dir ~/duke/temp/yolaatar/resinv_exp/data/TEM2/001350 \
    --original-px 0.00493 \
    --output-dir ~/duke/temp/yolaatar/resinv_exp/results_nnunet_tem2 \
    --gpu-id 0 \
    2>&1 | tee ~/output_eval_tem2_multires.log
```

---

## Process results on tassan (no need to pull prediction PNGs)

### Transfer processing scripts to tassan (on your Mac)

```bash
rsync -avz /Users/yolaatar/Developer/ADS/resinv/recompute_metrics.py yolaa@tassan.neuro.polymtl.ca:~/resinv_exp/scripts/
rsync -avz /Users/yolaatar/Developer/ADS/resinv/plot_resinv.py yolaa@tassan.neuro.polymtl.ca:~/resinv_exp/scripts/
```

### Install dependencies (one-time, on tassan)

```bash
source ~/resinv_exp/venv_resinv/bin/activate
pip install pandas matplotlib monai scikit-image -q
```

### Recompute metrics (run on joplin — duke accessible there)

```bash
source ~/resinv_exp/venv_resinv/bin/activate
DATA_TEM1="${HOME}/duke/temp/yolaatar/resinv_exp/data/TEM1"
DATA_TEM2="${HOME}/duke/temp/yolaatar/resinv_exp/data/TEM2/001350"

# TEM1 — models 3 and 4 only
python ~/resinv_exp/scripts/recompute_metrics.py \
    --results-dir ~/duke/temp/yolaatar/resinv_exp/results_nnunet \
    --data-dir ${DATA_TEM1} \
    --models da5 da5_multires

# TEM2 — models 3 and 4, GT subjects only (10 images: sub-370 s6/7, sub-372 s5, sub-373C s1c1/c2, sub-374 s4, sub-375 s1/5/6/7)
python ~/resinv_exp/scripts/recompute_metrics.py \
    --results-dir ~/duke/temp/yolaatar/resinv_exp/results_nnunet_tem2 \
    --data-dir ${DATA_TEM2} \
    --models da5 da5_multires \
    --gt-only

# ADS baseline (TEM1)
python ~/resinv_exp/scripts/recompute_metrics.py \
    --results-dir ~/duke/temp/yolaatar/resinv_exp/results_tem1 \
    --data-dir ${DATA_TEM1}
```

### Plot (run on joplin — duke accessible there)

```bash
source ~/resinv_exp/venv_resinv/bin/activate

python ~/resinv_exp/scripts/plot_resinv.py \
    --results-dir ~/duke/temp/yolaatar/resinv_exp/results_nnunet \
    --exclude-labels uaxon

python ~/resinv_exp/scripts/plot_resinv.py \
    --results-dir ~/duke/temp/yolaatar/resinv_exp/results_nnunet_tem2 \
    --exclude-labels uaxon
```

### Pull only CSVs and plots (on your Mac)

```bash
mkdir -p /Users/yolaatar/Developer/ADS/resinv/results_nnunet_tem2/{witness,multires} && ssh "yolaa@ge.polymtl.ca@joplin.neuro.polymtl.ca" "tar -C ~/duke/temp/yolaatar/resinv_exp/results_nnunet_tem2 -cf - witness/results.csv witness/results.png multires/results.csv multires/results.png comparison.png" | tar -C /Users/yolaatar/Developer/ADS/resinv/results_nnunet_tem2 -xf -
```

### Pull prediction images — GT subjects only (on your Mac)

GT subjects: sub-370, sub-372, sub-373C, sub-374, sub-375 (10 images total)

```bash
mkdir -p /Users/yolaatar/Developer/ADS/resinv/results_nnunet_tem2/{witness,multires} && ssh "yolaa@ge.polymtl.ca@joplin.neuro.polymtl.ca" "cd ~/duke/temp/yolaatar/resinv_exp/results_nnunet_tem2 && find witness multires -type f -name '*.png' \( -path '*/sub-370*' -o -path '*/sub-372*' -o -path '*/sub-373C*' -o -path '*/sub-374*' -o -path '*/sub-375*' \) | tar -cf - -T -" | tar -C /Users/yolaatar/Developer/ADS/resinv/results_nnunet_tem2 -xf -
```
