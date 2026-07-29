from pathlib import Path

import numpy as np
from PIL import Image

from pore_assessment.prepare import prepare_dataset


def _write_image(path: Path, color: tuple[int, int, int], offset: int) -> None:
    seed = color[0] * 10_000 + color[1] * 100 + offset
    pixels = np.random.default_rng(seed).integers(
        low=0,
        high=256,
        size=(32, 32, 3),
        dtype=np.uint8,
    )
    image = Image.fromarray(pixels, mode="RGB")
    image.save(path, quality=95)


def test_prepare_creates_disjoint_stratified_splits(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    for label in range(1, 6):
        class_dir = dataset_root / str(label)
        class_dir.mkdir(parents=True)
        for index in range(12):
            _write_image(
                class_dir / f"{label}_{index}.jpg",
                color=(label * 30, index * 10, 80),
                offset=index,
            )

    output_dir = tmp_path / "output"
    summary = prepare_dataset(
        dataset_root,
        output_dir,
        max_dhash_distance=0,
        seed=7,
    )

    assert summary["total_files"] == 60
    assert set(summary["split_class_counts"]) == {"train", "validation", "test"}
    for split_counts in summary["split_class_counts"].values():
        assert set(map(int, split_counts)) == {1, 2, 3, 4, 5}
    assert (output_dir / "audit.csv").is_file()
    assert (output_dir / "manifest.csv").is_file()
    assert (output_dir / "summary.json").is_file()


def test_cross_label_perceptual_conflict_is_excluded(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    for label in range(1, 6):
        class_dir = dataset_root / str(label)
        class_dir.mkdir(parents=True)
        for index in range(4):
            _write_image(
                class_dir / f"{label}_{index}.png",
                color=(label * 35, index * 30, 100),
                offset=index,
            )

    shared = Image.new("RGB", (32, 32), (120, 80, 60))
    shared.save(dataset_root / "1" / "shared.png")
    shared.save(dataset_root / "2" / "shared.png")

    summary = prepare_dataset(
        dataset_root,
        tmp_path / "output",
        max_dhash_distance=0,
    )

    assert summary["status_counts"]["label_conflict"] >= 2
