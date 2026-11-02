# Results

## Dice vs GT (mean across all pixel sizes, sub-373C excluded from TEM2)

| Model | TEM1 axon | TEM1 myelin | TEM2 axon | TEM2 myelin |
|---|---|---|---|---|
| witness | 0.446 | 0.418 | 0.448 | 0.423 |
| da5 | 0.450 | 0.423 | 0.429 | 0.408 |
| multires | 0.939 | 0.862 | 0.926 | 0.865 |
| da5_multires | 0.940 | 0.861 | 0.926 | 0.865 |

## DA5 trainer gain vs standard trainer

| Pair | TEM1 axon | TEM1 myelin | TEM2 axon | TEM2 myelin |
|---|---|---|---|---|
| witness vs da5 | +0.004 | +0.005 | -0.019 | -0.015 |
| multires vs da5_multires | +0.000 | -0.001 | +0.000 | +0.000 |

The DA5 trainer provides no meaningful improvement. The dominant factor is multi-resolution training.

## Morphometrics at native pixel size (10% sample)

| Dataset | Model | Mean axon diam (µm) | Mean g-ratio |
|---|---|---|---|
| TEM1 (2.36 nm/px) | witness | 0.628 | 0.732 |
| TEM1 (2.36 nm/px) | multires | 0.643 | 0.735 |
| TEM2 (4.93 nm/px) | witness | 0.621 | 0.733 |
| TEM2 (4.93 nm/px) | multires | 0.605 | 0.745 |

Multires produces stable morphometrics across all 16 pixel sizes. Witness degrades above ~7 nm (inflated diameter, unreliable g-ratio).
