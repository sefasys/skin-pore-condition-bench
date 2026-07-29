"""Create visual and quantitative checks for deterministic preprocessing."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from skimage import color

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from pore_assessment.preprocessing import (
    DEFAULT_CONFIG,
    PREPROCESSING_METHODS,
    DeterministicPreprocessing,
)

DISPLAY_NAMES = {
    "raw": "Raw RGB",
    "clahe": "Luminance CLAHE",
    "retinex": "Multi-scale Retinex",
    "adaptive_gamma": "Adaptive gamma",
}


def _resize(image: Image.Image, image_size: int) -> Image.Image:
    return image.convert("RGB").resize(
        (image_size, image_size),
        resample=Image.Resampling.BILINEAR,
    )


def _luminance(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    return color.rgb2lab(rgb)[..., 0] / 100.0


def audit(args: argparse.Namespace) -> pd.DataFrame:
    frame = pd.read_csv(args.manifest)
    frame = frame[frame["split"] == args.split].copy()
    if frame.empty:
        raise ValueError(f"No images found in split {args.split!r}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = {
        method: DeterministicPreprocessing(method) for method in PREPROCESSING_METHODS
    }
    records = []
    for row in frame.itertuples(index=False):
        path = args.dataset_root / row.path
        with Image.open(path) as source:
            resized = _resize(source, args.image_size)
        for method, transform in methods.items():
            started = time.perf_counter()
            transformed = transform(resized)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            luminance = _luminance(transformed)
            records.append(
                {
                    "method": method,
                    "path": row.path,
                    "label": int(row.label),
                    "elapsed_ms": elapsed_ms,
                    "luminance_mean": float(luminance.mean()),
                    "luminance_std": float(luminance.std()),
                    "near_black_fraction": float((luminance <= 0.01).mean()),
                    "near_white_fraction": float((luminance >= 0.99).mean()),
                }
            )

    detail = pd.DataFrame(records)
    detail.to_csv(args.output_dir / "per_image_statistics.csv", index=False)
    summary = (
        detail.groupby("method", sort=False)
        .agg(
            images=("path", "count"),
            elapsed_ms_mean=("elapsed_ms", "mean"),
            elapsed_ms_std=("elapsed_ms", "std"),
            luminance_mean=("luminance_mean", "mean"),
            luminance_std_mean=("luminance_std", "mean"),
            near_black_fraction=("near_black_fraction", "mean"),
            near_white_fraction=("near_white_fraction", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False)

    examples = (
        frame.sort_values(["label", "path"])
        .groupby("label", group_keys=False)
        .head(args.samples_per_class)
        .reset_index(drop=True)
    )
    figure, axes = plt.subplots(
        len(examples),
        len(methods),
        figsize=(12, max(3, 2.55 * len(examples))),
        squeeze=False,
    )
    for row_index, row in enumerate(examples.itertuples(index=False)):
        with Image.open(args.dataset_root / row.path) as source:
            resized = _resize(source, args.image_size)
        for column_index, (method, transform) in enumerate(methods.items()):
            axis = axes[row_index, column_index]
            axis.imshow(transform(resized))
            axis.axis("off")
            if row_index == 0:
                axis.set_title(DISPLAY_NAMES[method], fontsize=11)
            if column_index == 0:
                axis.set_ylabel(
                    f"Class {row.label}\n{row.label_name.replace('_', ' ').title()}",
                    fontsize=9,
                )
    figure.suptitle(
        f"Frozen preprocessing comparison — {args.split} examples",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.99))
    figure.savefig(
        args.output_dir / "comparison_grid.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    configuration = {
        "manifest": str(args.manifest.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "split": args.split,
        "image_size": args.image_size,
        "methods": list(PREPROCESSING_METHODS),
        "parameters": DEFAULT_CONFIG.to_dict(),
        "images_audited": int(len(frame)),
        "example_images": examples["path"].tolist(),
    }
    with (args.output_dir / "configuration.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(configuration, handle, indent=2)
        handle.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/data/manifest.csv"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/pore_data_set_224"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/preprocessing-audit"),
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation"),
        default="validation",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--samples-per-class", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    summary = audit(parse_args())
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
