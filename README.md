# Resolution Invariance Evaluation

Tests how stable ADS axon/myelin segmentation is when corpus callosum images
are downsampled from the original 0.0018625 μm/px to coarser resolutions.

Two models are benchmarked:
- **Cambridge unmyelinated TEM model** — trained at 0.00493 μm/px
- **Generalist model** (`model_seg_generalist_light`) — trained on a multi-resolution dataset

---

## Setup on lab GPU

```bash
# 1. Clone and install ADS
git clone https://github.com/axondeepseg/axondeepseg
cd axondeepseg
pip install -e ".[dev]"

# 2. Install extra dependencies
pip install scikit-image pandas matplotlib seaborn

# 3. Download models
download_model
```

Both models ship with ADS:
```
axondeepseg/AxonDeepSeg/models/model_seg_unmyelinated_stanford_light   # TEM unmyelinated
axondeepseg/AxonDeepSeg/models/model_seg_generalist_light              # generalist
```

---

## Running the evaluation

### Unmyelinated TEM model (trained at 0.00493 μm/px)

The primary label for this model is `uaxon` (unmyelinated axon).

```bash
python evaluate_resinv.py \
    --data-dir /path/to/timmler_nnunet/fullsizedata/Corpus_Callosum \
    --model-path /path/to/axondeepseg/AxonDeepSeg/models/model_seg_unmyelinated_stanford_light \
    --model-name unmyelinated_tem \
    --original-px 0.0018625 \
    --label uaxon \
    --secondary-label myelin \
    --output-dir ./results \
    --gpu-id 0
```

### Generalist model

```bash
python evaluate_resinv.py \
    --data-dir /path/to/timmler_nnunet/fullsizedata/Corpus_Callosum \
    --model-path /path/to/axondeepseg/AxonDeepSeg/models/model_seg_generalist_light \
    --model-name generalist \
    --original-px 0.0018625 \
    --label axon \
    --secondary-label myelin \
    --output-dir ./results \
    --gpu-id 0
```

Run both with the same `--output-dir` — the plot script will pick them both up.

### Plot (after both models have run)

```bash
python plot_resinv.py --results-dir ./results
```

---

## Pixel sizes tested (default)

| Pixel size (μm/px) | Scale vs original | Approx. image size |
|---|---|---|
| 0.0018625 | 1× (full res) | ~17655×12876 |
| 0.00493 | 0.38× — training resolution (reference) | ~6669×4863 |
| 0.007 | 0.27× | ~4700×3428 |
| 0.01 | 0.19× | ~3290×2400 |
| 0.013 | 0.14× | ~2530×1845 |
| 0.016 | 0.12× — maximum tested | ~2054×1498 |

All six resolutions produce images well above the 128 px minimum — no points
will be skipped for this dataset.

---

## Metrics

All metrics are computed relative to the model's prediction at the **training
resolution (0.00493 μm/px)**, which serves as the stability reference.

| Column | Description |
|---|---|
| `dice_axon_native` | Dice: prediction (resized to ref space) vs reference |
| `dice_axon_resized` | Dice: prediction vs reference (both at downsampled res) |
| `dice_axon_interp_baseline` | Dice: round-trip-resampled reference vs reference — how much interpolation alone degrades the signal |
| `dice_myelin_*` | Same three metrics for the myelin channel |

A model with perfect resolution invariance would have `dice_axon_native = 1.0`
at all pixel sizes. The interp baseline shows the ceiling: if round-trip
resampling already costs 0.05 Dice, you can't expect the model to do better.

---

## Output structure

```
results/
  cambridge_unmyelinated/
    IMAGE_NAME/
      downsampled/          # resized input images (PNG)
      predictions/          # ADS output masks
      metrics.csv           # per-resolution metrics for this image
    results.csv             # all images combined
    results.png             # per-model plot
  generalist/
    ...
    results.csv
    results.png
  comparison.png            # side-by-side model comparison
```

---

## Tips

- The script is resumable: downsampled images and predictions already on disk
  are not regenerated.
- To add more pixel sizes after a run, just add them to `--pixel-sizes` and
  re-run; only the new resolutions will be segmented.
- Pass `--gpu-id -1` to run on CPU (slow but useful for testing).
