# Resolution Invariance Evaluation

Tests how stable ADS segmentation is when images are downsampled across a range of pixel sizes. Takes images at full resolution, runs ADS at multiple scales (log-spaced from full-res to coarser resolutions), and measures Dice consistency against the prediction at the model's training resolution.

Two models benchmarked:
- **Unmyelinated TEM** (`model_seg_unmyelinated_stanford_light`) — trained at 0.00493 μm/px
- **Generalist** (`model_seg_generalist_light`) — multi-resolution dataset

---

## Running

### Unmyelinated TEM model

```bash
CUDA_VISIBLE_DEVICES="0" python evaluate_resinv.py \
    --data-dir /path/to/images \
    --model-path /path/to/model_seg_unmyelinated_stanford_light \
    --model-name unmyelinated_tem \
    --label uaxon \
    --secondary-label myelin \
    --output-dir ./results \
    --gpu-id 0 \
    --crop-size 4096
```

### Generalist model

```bash
CUDA_VISIBLE_DEVICES="0" python evaluate_resinv.py \
    --data-dir /path/to/images \
    --model-path /path/to/model_seg_generalist_light \
    --model-name generalist \
    --label axon \
    --secondary-label myelin \
    --output-dir ./results \
    --gpu-id 0 \
    --crop-size 4096
```

Run both with the same `--output-dir`. Use `--gpu-id -1` for CPU.

### Plot

```bash
python plot_resinv.py --results-dir ./results
```

---

## Output

```
results/
  <model-name>/
    <image>/
      downsampled/      # resized inputs
      predictions/      # ADS masks
      metrics.csv
    results.csv
    results.png
  comparison.png
```

The script is resumable: existing downsampled images and predictions are not regenerated. To test extra pixel sizes, add them to `--pixel-sizes` and re-run.
