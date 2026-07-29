# Ethics, Intended Use, and Limitations

## Intended use

This software is an experimental research pipeline for comparing deterministic
image preprocessing in an ordinal computer-vision task. It may support
education, reproducibility work, and hypothesis generation.

## Not intended for

- clinical diagnosis, screening, triage, or treatment;
- estimation of dry, oily, normal, or combination skin type;
- inference of hydration, sebum, disease, or general skin health;
- autonomous patient-facing recommendations;
- performance claims about populations not represented and independently
  evaluated.

The labels are dataset-specific visual pore-condition categories. A model can
learn capture conditions, devices, lighting, post-processing, or annotation
style rather than medically meaningful features.

## Privacy and data governance

Facial skin images may be sensitive even when tightly cropped. Keep restricted
data in controlled storage, minimize access, and follow the original consent,
data-use agreement, institutional review, and applicable privacy law. Do not
commit image files, prediction tables containing identifying filenames, or
example grids to a public repository.

Before collecting new participant images, define a real scientific question,
consent language, retention policy, deletion process, security controls, and
qualified ethical/clinical oversight. This experimental model alone is not a
justification for collecting patient data.

## Major limitations

- small, single-source dataset;
- no verified subject identifiers for subject-disjoint splitting;
- unknown demographic, camera, lighting, and acquisition coverage;
- possible remaining visually related images despite perceptual hashing;
- ordinal labels are not independently re-annotated;
- only three random seeds;
- validation data was used for model and method selection;
- no held-out test result yet;
- no external, prospective, or clinical validation;
- no subgroup fairness or domain-shift evaluation.

Any future user interface should display uncertainty and scope limitations and
must not imply medical clearance or diagnosis.

