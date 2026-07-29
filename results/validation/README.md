# Published Validation Artifacts

This directory contains the source-image-free aggregate artifacts from the
frozen ConvNeXt-Tiny preprocessing comparison.

- `per_seed_metrics.csv`: one validation row per method and seed.
- `per_seed_class_metrics.csv`: precision, recall, and F1 by class.
- `method_summary.csv`: mean, sample SD, minimum, and maximum.
- `method_class_summary.csv`: class-level mean and sample SD.
- `paired_vs_raw_per_seed.csv`: paired method-minus-raw changes and exact
  McNemar tests.
- `cluster_bootstrap_vs_raw.csv`: cluster-bootstrap percentile intervals.
- `runtime_per_run.csv` and `runtime_totals.csv`: hardware-specific execution
  records.
- `environment.json`: frozen experiment and runtime metadata.
- `cache_metadata/`: parameters and lossless-cache verification records.
- `figures/`: aggregate plots with no source images.

These are **validation** results. The held-out test split was not evaluated.
Raw predictions are excluded because they contain filenames from a restricted
dataset. Checkpoints and cached images are also excluded.

