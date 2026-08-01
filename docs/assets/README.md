# Visual Asset Provenance

`synthetic_skin_source.png` was generated with OpenAI's built-in image
generation tool on 2026-08-01. It represents an anonymous, entirely synthetic
skin texture; it is not a sample from the restricted Zenodo dataset and does
not depict a study participant.

The generation request specified a neutral dermatology-style macro texture
with subtle pores, diffuse lighting, no identifiable facial structures, no
lesions, no text, and no watermark.

`preprocessing_comparison.png` is derived from that single source by
`scripts/generate_preprocessing_demo.py`. The four panels are produced by the
same `DeterministicPreprocessing` implementation used in the experiment:

1. raw RGB;
2. luminance CLAHE;
3. luminance multi-scale Retinex;
4. adaptive gamma correction.

Regenerate the comparison from the repository root with:

```bash
python scripts/generate_preprocessing_demo.py
```

The figure is explanatory only. It is not evidence of model performance and
should not be used to infer how every real skin image will respond.

