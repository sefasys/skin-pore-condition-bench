# Methodology

## Question

Does fixed, deterministic contrast or illumination preprocessing improve
five-class ordinal pore-condition classification when architecture, split,
augmentation, optimization, and random seeds are held constant?

## Input methods

All images are resized to 224 × 224 before deterministic preprocessing.

- **Raw RGB:** no enhancement.
- **CLAHE:** contrast-limited adaptive histogram equalization on CIELAB
  luminance (`clip_limit=0.01`, `kernel_size=28`). Chroma is retained.
- **Multi-scale Retinex:** log-luminance responses at Gaussian scales
  `[15, 60, 180]`, averaged and rescaled using the 1st and 99th percentiles.
  Chroma is retained.
- **Adaptive gamma:** a weighted luminance-CDF gamma correction with weight
  `0.5`. Chroma is retained.

Enhanced images are cached as PNG and read back to verify pixel-identical,
lossless storage. Cache metadata records the manifest hash and preprocessing
parameters. This avoids repeating Retinex on every epoch.

## Augmentation

Training uses horizontal flips, vertical flips, and rotations of 0°, 90°, 180°,
or 270°. Right-angle rotation avoids interpolated borders. Evaluation is
deterministic and has no stochastic augmentation.

## Model and optimization

- ConvNeXt-Tiny with ImageNet-1K pretrained weights
- Five-class linear classifier head
- Cross entropy with square-root inverse-frequency class weights
- Label smoothing: 0.05
- AdamW
- Classifier-head learning rate: `3e-4`
- Backbone learning rate: `3e-5`
- Weight decay: `1e-4`
- Reduce-on-plateau scheduler: factor `0.3`, patience `2`
- Maximum epochs: 35
- Early stopping patience: 7
- Batch size: 32
- Best-checkpoint selection: validation exact accuracy
- Paired seeds: 42, 123, 2026

The same recipe is used for every preprocessing method.

## Evaluation

The primary metric is exact accuracy. Complementary metrics include macro F1,
balanced accuracy, quadratic weighted kappa (QWK), ordinal mean absolute error,
errors of two or more classes, calibration error, macro one-vs-rest ROC AUC,
average precision, per-class metrics, confusion matrices, and learning curves.

Paired-seed comparisons report method-minus-raw differences and an exact
McNemar test for prediction correctness. A cluster bootstrap resamples
perceptual split groups and reports 95% percentile intervals for exact
accuracy, ordinal MAE, and severe-error differences.

With only three seeds, seed-level uncertainty remains imprecise. Bootstrap
intervals account for clustered validation images but do not solve external
validity, labeling, or dataset-source uncertainty.

## Model-selection firewall

All architecture and preprocessing choices are made using train and validation
data. The final test is locked in both the reporting command and the
orchestrator. It requires the explicit `--evaluate-test` flag and should be
evaluated only after code, configuration, and interpretation are frozen.

