# Reproducibility

## Frozen protocol

The machine-readable final protocol is
[`configs/final_protocol.json`](../configs/final_protocol.json). Change it only
for a new, separately named experiment. Do not overwrite the reported
validation study.

## Recommended sequence

1. Obtain the restricted dataset and place it under
   `data/pore_data_set_224/{1,2,3,4,5}`.
2. Install the project in an isolated Python environment.
3. Run `pytest`.
4. Preview the orchestration with `pore-pipeline --dry-run`.
5. Run prepare, audit, and cache.
6. Train every method with the same paired seeds.
7. Generate validation reports and the aggregate summary.
8. Freeze the repository commit, configuration, and conclusions.
9. Only then run the explicit held-out test command once.

## Determinism

Python, NumPy, and PyTorch random generators are seeded. CUDA deterministic
algorithms and cuDNN deterministic mode are enabled by the trainer. Full
bit-for-bit identity can still depend on hardware, CUDA, PyTorch, image-library
versions, and pretrained-weight versions. Report all environment versions and
the commit hash with new results.

## Caches and resumption

The cache records method parameters, image size, manifest SHA-256, output
format, and lossless verification. Training saves `last.pt` with optimizer and
scheduler state every epoch. Use `--resume` to continue a pre-empted run.

The pipeline skips completed checkpoints and reports by default. `--force`
regenerates selected outputs; use it carefully because it may replace local
experimental artifacts.

## Colab

The notebook is intentionally thin. It installs and invokes this package rather
than duplicating training code in notebook cells. Mount Drive, set the
repository URL and dataset path, then run the cells from top to bottom.

The original experiment used PyTorch `2.11.0+cu128`, torchvision
`0.26.0+cu128`, ImageNet-1K ConvNeXt-Tiny V1 weights, and an NVIDIA A100-SXM4
80GB. These versions describe the run; they are not guaranteed minimums.

