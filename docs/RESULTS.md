# Validation Results

These are validation results, not held-out test or external-validation results.

## Three-seed summary

| Method | Accuracy mean ± SD | Macro F1 | Balanced accuracy | QWK | MAE | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Adaptive gamma | **0.774 ± 0.023** | **0.798** | **0.802** | **0.903** | **0.248** | 0.073 |
| Raw RGB | 0.750 ± 0.017 | 0.771 | 0.775 | 0.897 | 0.271 | 0.068 |
| Retinex | 0.738 ± 0.037 | 0.757 | 0.755 | 0.893 | 0.282 | 0.076 |
| CLAHE | 0.731 ± 0.032 | 0.753 | 0.754 | 0.893 | 0.290 | **0.059** |

All metrics are means across seeds 42, 123, and 2026 except the explicitly
shown sample standard deviation.

## Adaptive gamma versus raw RGB

- Mean exact-accuracy difference: +0.0242
- Per-seed accuracy differences: +0.0121, −0.0060, +0.0665
- Cluster-bootstrap 95% interval: [−0.0010, +0.0493]
- Mean QWK difference: +0.0057
- Mean ordinal-MAE difference: −0.0232
- Severe-error-rate difference: 0.0000

Adaptive gamma ranked first on mean accuracy, macro F1, balanced accuracy, QWK,
and ordinal MAE. However, one seed was slightly worse than raw and the
bootstrap accuracy interval crossed zero. The defensible conclusion is that
adaptive gamma is a promising candidate under this protocol, not that it has
demonstrated universal or statistically decisive superiority.

Retinex showed the greatest seed variability and was the slowest preprocessing
step. CLAHE achieved the lowest mean expected calibration error but reduced
mean classification performance.

## Runtime

The reported Google Colab A100 execution recorded:

| Stage | Time |
|---|---:|
| Lossless preprocessing cache | 504.9 s |
| Training (12 runs) | 1,922.7 s |
| Validation reports | 187.5 s |
| Total | approximately 43.6 min |

Retinex cache construction alone took 375.5 seconds. Runtime metadata is
hardware-specific and should not be interpreted as a benchmark.

## Artifacts

The `results/validation/` directory includes per-seed metrics, class metrics,
method summaries, paired comparisons, cluster-bootstrap intervals, runtime
records, cache metadata, and source-image-free figures. Checkpoints,
predictions, restricted image montages, and raw dataset files are deliberately
excluded.

