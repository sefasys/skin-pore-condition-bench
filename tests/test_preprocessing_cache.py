from argparse import Namespace

import numpy as np
import pandas as pd
from PIL import Image

from pore_assessment.preprocessing import DeterministicPreprocessing
from pore_assessment.preprocessing_cache import (
    build_cache,
    cache_image_root,
    cached_relative_path,
    validate_cache,
)


def test_retinex_cache_is_lossless_and_matches_on_the_fly_output(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    source_path = dataset_root / "1" / "example.jpg"
    source_path.parent.mkdir(parents=True)
    y, x = np.mgrid[0:32, 0:32]
    pixels = (
        np.stack(
            (
                70 + x * 3,
                55 + y * 3,
                45 + ((x + y) % 20),
            ),
            axis=-1,
        )
        .clip(0, 255)
        .astype(np.uint8)
    )
    Image.fromarray(pixels, mode="RGB").save(source_path, quality=95)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "path": "1/example.jpg",
                "label": 1,
                "label_name": "very_good",
                "split": "train",
            }
        ]
    ).to_csv(manifest, index=False)
    cache_root = tmp_path / "cache"

    results = build_cache(
        Namespace(
            manifest=manifest,
            dataset_root=dataset_root,
            cache_root=cache_root,
            methods=["retinex"],
            image_size=64,
        )
    )

    assert results[0]["lossless_png_verified"] is True
    validate_cache(cache_root, "retinex", manifest, 64)
    cached_path = cache_image_root(cache_root, "retinex") / cached_relative_path(
        "1/example.jpg"
    )
    with Image.open(source_path) as source:
        resized = source.convert("RGB").resize(
            (64, 64),
            resample=Image.Resampling.BILINEAR,
        )
    expected = DeterministicPreprocessing("retinex")(resized)
    with Image.open(cached_path) as cached:
        actual = cached.convert("RGB")
    assert np.array_equal(np.asarray(expected), np.asarray(actual))


def test_cache_reuses_completed_lossless_files(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    source_path = dataset_root / "2" / "example.png"
    source_path.parent.mkdir(parents=True)
    Image.new("RGB", (16, 16), (100, 80, 60)).save(source_path)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([{"path": "2/example.png", "label": 2, "split": "validation"}]).to_csv(
        manifest, index=False
    )
    cache_root = tmp_path / "cache"
    args = Namespace(
        manifest=manifest,
        dataset_root=dataset_root,
        cache_root=cache_root,
        methods=["adaptive_gamma"],
        image_size=16,
    )

    first = build_cache(args)[0]
    second = build_cache(args)[0]

    assert first["created_this_run"] == 1
    assert second["created_this_run"] == 0
    assert second["reused_this_run"] == 1


def test_cache_rejects_reuse_when_manifest_changes(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    source_path = dataset_root / "1" / "example.png"
    source_path.parent.mkdir(parents=True)
    Image.new("RGB", (16, 16), (100, 80, 60)).save(source_path)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([{"path": "1/example.png", "label": 1, "split": "train"}]).to_csv(
        manifest, index=False
    )
    args = Namespace(
        manifest=manifest,
        dataset_root=dataset_root,
        cache_root=tmp_path / "cache",
        methods=["clahe"],
        image_size=16,
    )
    build_cache(args)
    pd.DataFrame([{"path": "1/example.png", "label": 1, "split": "validation"}]).to_csv(
        manifest, index=False
    )

    try:
        build_cache(args)
    except ValueError as error:
        assert "metadata mismatch" in str(error)
    else:
        raise AssertionError("A cache with a stale manifest hash was accepted")
