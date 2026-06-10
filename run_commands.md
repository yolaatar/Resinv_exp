# Run commands

## Test single image inference

```bash
CUDA_VISIBLE_DEVICES="0" python ~/resinv_exp/scripts/evaluate_resinv.py --data-dir ~/duke/temp/yolaatar/resinv_exp/data/Corpus_Callosum --model-path ~/axondeepseg/AxonDeepSeg/models/model_seg_unmyelinated_stanford_light --model-name unmyelinated_tem --label uaxon --secondary-label myelin --output-dir ~/duke/temp/yolaatar/resinv_exp/results --gpu-id 0 --crop-size 4096 --images ALIC_CC_II_01_P202503101408 2>&1 | tee ~/output.log
```

## Clean previous results

```bash
rm -rf ~/duke/temp/yolaatar/resinv_exp/results/
```

## Unmyelinated TEM model

```bash
CUDA_VISIBLE_DEVICES="0" python ~/resinv_exp/scripts/evaluate_resinv.py --data-dir ~/duke/temp/yolaatar/resinv_exp/data/Corpus_Callosum --model-path ~/axondeepseg/AxonDeepSeg/models/model_seg_unmyelinated_stanford_light --model-name unmyelinated_tem --label uaxon --secondary-label myelin --output-dir ~/duke/temp/yolaatar/resinv_exp/results --gpu-id 0 --crop-size 4096 2>&1 | tee ~/output.log
```

## Generalist model

```bash
CUDA_VISIBLE_DEVICES="0" python ~/resinv_exp/scripts/evaluate_resinv.py \
    --data-dir ~/duke/temp/yolaatar/resinv_exp/data/Corpus_Callosum \
    --model-path ~/axondeepseg/AxonDeepSeg/models/model_seg_generalist_light \
    --model-name generalist \
    --label axon --secondary-label myelin \
    --output-dir ~/duke/temp/yolaatar/resinv_exp/results \
    --gpu-id 0 \
    --crop-size 4096
```

## Retrieve results (on your Mac)

```bash
rsync -avz yolaa@tassan.neuro.polymtl.ca:~/duke/temp/yolaatar/resinv_exp/results/ \
    /Users/yolaatar/Developer/ADS/resinv/results/
```

## Plot

```bash
cd /Users/yolaatar/Developer/ADS/resinv
source ../axondeepseg/.venv/bin/activate
python plot_resinv.py --results-dir ./results
```
