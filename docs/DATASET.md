# Dataset

## Source

This project was developed against:

- Čedomir Vasić (2023), *Dataset used to train a Convolutional Neural Network
  dedicated to skin pore detection and classification*.
- Zenodo record: <https://zenodo.org/records/8228942>
- DOI: <https://doi.org/10.5281/zenodo.8228942>

The Zenodo files are restricted. The record states that access may be granted
on reasonable request for scientific research. Obtain the files directly from
the owner and retain the approval record.

## Redistribution

No images are included in this repository. Do not upload the images, derived
face montages, or redistributed archives to GitHub, Kaggle, Hugging Face, or
another service unless the data owner explicitly authorizes redistribution in
writing. Repository code licensing does not grant rights to the dataset.

## Expected layout

```text
data/pore_data_set_224/
├── 1/
├── 2/
├── 3/
├── 4/
└── 5/
```

| Directory | Ordinal label |
|---:|---|
| 1 | very_good |
| 2 | good |
| 3 | normal |
| 4 | poor |
| 5 | very_poor |

These names describe the source dataset's pore-condition scale. They should not
be reinterpreted as dry, oily, normal, or combination skin.

## Local audit

The committed aggregate audit is:

| Item | Count |
|---|---:|
| Discovered images | 3,086 |
| Excluded duplicates | 851 |
| Excluded label conflicts | 10 |
| Retained images | 2,225 |
| Train | 1,559 |
| Validation | 331 |
| Test | 335 |

Retained class counts were 427, 545, 580, 412, and 261 for classes 1 through 5.
The pipeline regenerates `artifacts/data/audit.csv`, `manifest.csv`, and
`summary.json` locally. These files are ignored because they may expose
restricted filenames and local paths.

## Leakage controls

Each image receives a rotation-canonical 64-bit difference hash. Images with
the same canonical hash are treated as perceptual duplicates: label conflicts
are excluded, and one representative is retained from consistent groups.
Distinct hashes within Hamming distance 2 are retained but assigned a shared
split group. Group-aware splitting prevents those near-neighbours from crossing
train, validation, and test boundaries.

Perceptual hashing is a pragmatic safeguard, not proof that every related image
or subject has been identified. Subject identifiers were not available, so a
strict subject-level split could not be verified.

