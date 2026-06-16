# Run commands

## Test single image inference

```bash
CUDA_VISIBLE_DEVICES="0" python ~/resinv_exp/scripts/evaluate_resinv.py --data-dir ~/duke/temp/yolaatar/resinv_exp/data/Corpus_Callosum --model-path ~/axondeepseg/AxonDeepSeg/models/model_seg_unmyelinated_stanford_light --model-name unmyelinated_tem --label uaxon --secondary-label myelin --output-dir ~/duke/temp/yolaatar/resinv_exp/results --gpu-id 0 --crop-size 4096 --images ALIC_CC_II_01_P202503101408 2>&1 | tee ~/output.log
```

## Test single image inference (generalist)

```bash
CUDA_VISIBLE_DEVICES="0" python ~/resinv_exp/scripts/evaluate_resinv.py --data-dir ~/duke/temp/yolaatar/resinv_exp/data/Corpus_Callosum --model-path ~/axondeepseg/AxonDeepSeg/models/model_seg_generalist_light --model-name generalist --label axon --secondary-label myelin --output-dir ~/duke/temp/yolaatar/resinv_exp/results --gpu-id 0 --crop-size 4096 --images ALIC_CC_II_01_P202503101408 2>&1 | tee ~/output.log
```

## Clean previous results

```bash
rm -rf ~/duke/temp/yolaatar/resinv_exp/results/
```

## Run both models (sequential, GPU 0)

```bash
CUDA_VISIBLE_DEVICES="0" python ~/resinv_exp/scripts/evaluate_resinv.py --data-dir ~/duke/temp/yolaatar/resinv_exp/data/Corpus_Callosum --model-path ~/axondeepseg/AxonDeepSeg/models/model_seg_unmyelinated_stanford_light --model-name unmyelinated_tem --label uaxon --secondary-label myelin --output-dir ~/duke/temp/yolaatar/resinv_exp/results --gpu-id 0 --crop-size 4096 2>&1 | tee ~/output_unmyelinated.log && CUDA_VISIBLE_DEVICES="0" python ~/resinv_exp/scripts/evaluate_resinv.py --data-dir ~/duke/temp/yolaatar/resinv_exp/data/Corpus_Callosum --model-path ~/axondeepseg/AxonDeepSeg/models/model_seg_generalist_light --model-name generalist --label axon --secondary-label myelin --output-dir ~/duke/temp/yolaatar/resinv_exp/results --gpu-id 0 --crop-size 4096 2>&1 | tee ~/output_generalist.log
```

## Unmyelinated TEM model only

```bash
CUDA_VISIBLE_DEVICES="0" python ~/resinv_exp/scripts/evaluate_resinv.py --data-dir ~/duke/temp/yolaatar/resinv_exp/data/Corpus_Callosum --model-path ~/axondeepseg/AxonDeepSeg/models/model_seg_unmyelinated_stanford_light --model-name unmyelinated_tem --label uaxon --secondary-label myelin --output-dir ~/duke/temp/yolaatar/resinv_exp/results --gpu-id 0 --crop-size 4096 2>&1 | tee ~/output_unmyelinated.log
```

## Generalist model only

```bash
CUDA_VISIBLE_DEVICES="0" python ~/resinv_exp/scripts/evaluate_resinv.py --data-dir ~/duke/temp/yolaatar/resinv_exp/data/Corpus_Callosum --model-path ~/axondeepseg/AxonDeepSeg/models/model_seg_generalist_light --model-name generalist --label axon --secondary-label myelin --output-dir ~/duke/temp/yolaatar/resinv_exp/results --gpu-id 0 --crop-size 4096 2>&1 | tee ~/output_generalist.log
```

## Fix PyTorch Blackwell (sm_120) — tassan only

PyTorch cu124 stable doesn't include sm_120 kernels. Install nightly cu128 which adds Blackwell support:

```bash
pip uninstall torchvision -y && pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --upgrade
```

Verify sm_120 is now listed:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.get_arch_list())"
```

---

## Retrieve results — Corpus Callosum (on your Mac)

```bash
rsync -avz yolaa@tassan.neuro.polymtl.ca:~/duke/temp/yolaatar/resinv_exp/results/ /Users/yolaatar/Developer/ADS/resinv/results/
```

---

## TEM1 dataset (0.00236 μm/px, 158 images, GT available)

### Transfer data to tassan (on your Mac)

```bash
rsync -avz /Users/yolaatar/Developer/ADS/data/TEM1/ yolaa@tassan.neuro.polymtl.ca:~/duke/temp/yolaatar/resinv_exp/data/TEM1/
```

### Setup on tassan (first time only)

```bash
curl -o ~/resinv_exp/scripts/evaluate_resinv.py https://raw.githubusercontent.com/yolaatar/Resinv_exp/main/evaluate_resinv.py && pip install monai -q
```

### Run TEM1 — unmyelinated TEM model (on tassan)

```bash
CUDA_VISIBLE_DEVICES="0" python ~/resinv_exp/scripts/evaluate_resinv.py --data-dir ~/duke/temp/yolaatar/resinv_exp/data/TEM1 --model-path ~/axondeepseg/AxonDeepSeg/models/model_seg_unmyelinated_stanford_light --model-name unmyelinated_tem --label uaxon --secondary-label myelin --original-px 0.00236 --output-dir ~/duke/temp/yolaatar/resinv_exp/results_tem1 --gpu-id 0 2>&1 | tee ~/output_tem1.log
```

### Retrieve TEM1 results (on your Mac)

```bash
rsync -avz yolaa@tassan.neuro.polymtl.ca:~/duke/temp/yolaatar/resinv_exp/results_tem1/ /Users/yolaatar/Developer/ADS/resinv/results_tem1/
```

---

## Recompute metrics (local, no GPU)

### TEM1 — axon/myelin vs GT, uaxon vs prediction at training res
```bash
cd /Users/yolaatar/Developer/ADS/resinv
source ../axondeepseg/.venv/bin/activate
python recompute_metrics.py --results-dir ./results_tem1 --data-dir /Users/yolaatar/Developer/ADS/data/TEM1
```

### Corpus Callosum — no GT, all labels vs prediction at training res
```bash
cd /Users/yolaatar/Developer/ADS/resinv
source ../axondeepseg/.venv/bin/activate
python recompute_metrics.py --results-dir ./results
```

## Plot

```bash
cd /Users/yolaatar/Developer/ADS/resinv
source ../axondeepseg/.venv/bin/activate
python plot_resinv.py --results-dir ./results
```

---

## Training experiment (nnunetv2==2.2.1)

### Transfer scripts to tassan (on your Mac)

```bash
rsync -avz /Users/yolaatar/Developer/ADS/resinv/training/ yolaa@tassan.neuro.polymtl.ca:~/resinv_exp/scripts/training/
```

### Fix nnUNet 2.2.1 + PyTorch 2.10 incompatibility (one-time, on tassan)

**Patch 1 — polylr verbose arg (needed for training):**
```bash
sed -i 's/super().__init__(optimizer, current_step if current_step is not None else -1, False)/super().__init__(optimizer, current_step if current_step is not None else -1)/' ~/resinv_exp/venv_resinv/lib/python3.12/site-packages/nnunetv2/training/lr_scheduler/polylr.py
```

Verify:
```bash
grep "super().__init__" ~/resinv_exp/venv_resinv/lib/python3.12/site-packages/nnunetv2/training/lr_scheduler/polylr.py
```

**Patch 2 — weights_only in trainer (needed for training resume):**
```bash
sed -i "s/torch.load(filename_or_checkpoint, map_location=self.device)/torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)/" ~/resinv_exp/venv_resinv/lib/python3.12/site-packages/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py
```

**Patch 3 — weights_only in predictor (needed for inference):**
```bash
sed -i "s/torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),/torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name), weights_only=False,/" ~/resinv_exp/venv_resinv/lib/python3.12/site-packages/nnunetv2/inference/predict_from_raw_data.py
```

Verify:
```bash
grep "weights_only" ~/resinv_exp/venv_resinv/lib/python3.12/site-packages/nnunetv2/inference/predict_from_raw_data.py
```

### Model 1 — Witness (standard nnUNet, single resolution)

```bash
bash ~/resinv_exp/scripts/training/train_witness.sh
```

Logs: `~/output_witness.log`
Checkpoint: `/tmp/yolaatar/nnunet_results/Dataset001_TEM_witness/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_best.pth`

### Save witness model results from /tmp to duke (before session ends)

```bash
rsync -avz /tmp/yolaatar/nnunet_results/ ~/duke/temp/yolaatar/nnunet_resinv/nnUNet_results/
```

### Model 2 — Multi-resolution (standard nnUNet, 4x resolutions)

```bash
bash ~/resinv_exp/scripts/training/train_multires.sh
```

Logs: `~/output_multires.log`
Checkpoint: `~/duke/temp/yolaatar/nnunet_resinv/nnUNet_results/Dataset002_TEM_multires/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_best.pth`

### Model 3 — DA5 (nnUNetTrainerDA5, single resolution, same dataset as witness)

```bash
bash ~/resinv_exp/scripts/training/train_da5.sh
```

Logs: `~/output_da5.log`
Checkpoint: `~/duke/temp/yolaatar/nnunet_resinv/nnUNet_results/Dataset001_TEM_witness/nnUNetTrainerDA5__nnUNetPlans__2d/fold_0/checkpoint_best.pth`

### Model 4 — DA5 + Multi-resolution (nnUNetTrainerDA5, 4x resolutions)

```bash
bash ~/resinv_exp/scripts/training/train_da5_multires.sh
```

Logs: `~/output_da5_multires.log`
Checkpoint: `~/duke/temp/yolaatar/nnunet_resinv/nnUNet_results/Dataset002_TEM_multires/nnUNetTrainerDA5__nnUNetPlans__2d/fold_0/checkpoint_best.pth`

---

## Evaluation — nnUNet models on TEM1 test set

### Transfer scripts (on your Mac)

```bash
rsync -avz /Users/yolaatar/Developer/ADS/resinv/training/ yolaa@tassan.neuro.polymtl.ca:~/resinv_exp/scripts/training/
```

### Evaluate models 1 and 2 (GPU 1, on tassan)

```bash
source ~/resinv_exp/venv_resinv/bin/activate
bash ~/resinv_exp/scripts/training/run_evaluation.sh
```

Logs: `~/output_eval_witness.log`, `~/output_eval_multires.log`

### Evaluate models 3 and 4 (after training finishes)

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

### Retrieve results (on your Mac)

```bash
rsync -avz yolaa@tassan.neuro.polymtl.ca:~/duke/temp/yolaatar/resinv_exp/results_nnunet/ /Users/yolaatar/Developer/ADS/resinv/results_nnunet/
```

### Recompute metrics (local, no GPU)

```bash
cd /Users/yolaatar/Developer/ADS/resinv
source ../axondeepseg/.venv/bin/activate
python recompute_metrics.py --results-dir ./results_nnunet --data-dir /Users/yolaatar/Developer/ADS/data/TEM1
```
