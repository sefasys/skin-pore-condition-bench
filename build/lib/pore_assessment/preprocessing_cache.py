"""Build and validate lossless caches for deterministic preprocessing."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from pore_assessment.preprocessing import (
    DEFAULT_CONFIG,
    PREPROCESSING_METHODS,
    DeterministicPreprocessing,
)

CACHEABLE_METHODS = tuple(method for method in PREPROCESSING_METHODS if method != "raw")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cached_relative_path(source_path: str) -> Path:
    return Path(source_path).with_suffix(".png")


def cache_image_root(cache_root: Path, method: str) -> Path:
    return cache_root / method / "images"


def validate_cache(
    cache_root: Path,
    method: str,
    manifest: Path,
    image_size: int,
) -> dict:
    metadata_path = cache_root / method / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Preprocessing cache metadata not found: {metadata_path}"
        )
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected = {
        "method": method,
        "image_size": image_size,
        "manifest_sha256": sha256_file(manifest),
        "parameters": DEFAULT_CONFIG.to_dict(),
        "lossless_png_verified": True,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Preprocessing cache metadata mismatch: {mismatches}")
    return metadata


def build_method_cache(
    frame: pd.DataFrame,
    dataset_root: Path,
    cache_root: Path,
    manifest: Path,
    method: str,
    image_size: int,
) -> dict:
    if method not in CACHEABLE_METHODS:
        raise ValueError(f"Cache method must be one of {CACHEABLE_METHODS}")
    transform = DeterministicPreprocessing(method)
    image_root = cache_image_root(cache_root, method)
    metadata_path = cache_root / method / "metadata.json"
    has_cached_files = image_root.is_dir() and any(image_root.rglob("*.png"))
    if has_cached_files:
        if not metadata_path.is_file():
            raise FileNotFoundError(
                "Cached PNG files exist without metadata; remove or rebuild "
                f"the method cache: {image_root}"
            )
        validate_cache(cache_root, method, manifest, image_size)
    image_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    created = 0
    reused = 0

    for index, row in enumerate(frame.itertuples(index=False), start=1):
        destination = image_root / cached_relative_path(row.path)
        if destination.is_file():
            reused += 1
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(dataset_root / row.path) as source:
                resized = source.convert("RGB").resize(
                    (image_size, image_size),
                    resample=Image.Resampling.BILINEAR,
                )
            transformed = transform(resized)
            temporary = destination.with_suffix(".png.part")
            transformed.save(temporary, format="PNG", compress_level=1)
            with Image.open(temporary) as restored:
                restored.load()
                if not np.array_equal(
                    np.asarray(transformed),
                    np.asarray(restored.convert("RGB")),
                ):
                    raise RuntimeError(
                        f"Lossless cache verification failed: {destination}"
                    )
            temporary.replace(destination)
            created += 1
        if index % 100 == 0 or index == len(frame):
            print(
                f"{method}: cached {index}/{len(frame)} images "
                f"(created={created}, reused={reused})",
                flush=True,
            )

    missing = [
        row.path
        for row in frame.itertuples(index=False)
        if not (image_root / cached_relative_path(row.path)).is_file()
    ]
    if missing:
        raise RuntimeError(f"Cache is missing {len(missing)} images")
    metadata = {
        "method": method,
        "image_size": image_size,
        "images": int(len(frame)),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "dataset_root": str(dataset_root.resolve()),
        "parameters": DEFAULT_CONFIG.to_dict(),
        "format": "PNG",
        "lossless_png_verified": True,
        "created_this_run": created,
        "reused_this_run": reused,
        "elapsed_seconds": time.perf_counter() - started,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    return metadata


def build_cache(args: argparse.Namespace) -> list[dict]:
    frame = pd.read_csv(args.manifest)
    if "path" not in frame:
        raise ValueError("Manifest must contain a path column")
    if frame["path"].duplicated().any():
        raise ValueError("Manifest paths must be unique")
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")
    args.cache_root.mkdir(parents=True, exist_ok=True)
    return [
        build_method_cache(
            frame=frame,
            dataset_root=args.dataset_root,
            cache_root=args.cache_root,
            manifest=args.manifest,
            method=method,
            image_size=args.image_size,
        )
        for method in args.methods
    ]


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
        "--cache-root",
        type=Path,
        default=Path("artifacts/preprocessing-cache"),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=CACHEABLE_METHODS,
        default=["retinex"],
    )
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    results = build_cache(parse_args())
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
