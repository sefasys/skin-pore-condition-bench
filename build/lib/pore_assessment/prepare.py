"""Audit, deduplicate, and split the five-class Zenodo pore dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

CLASS_NAMES = {
    1: "very_good",
    2: "good",
    3: "normal",
    4: "poor",
    5: "very_poor",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class ImageRecord:
    path: str
    label: int
    label_name: str
    width: int | None = None
    height: int | None = None
    mode: str | None = None
    sha256: str | None = None
    dhash: str | None = None
    duplicate_group_id: str | None = None
    split_group_id: str | None = None
    status: str = "unprocessed"
    split: str | None = None
    error: str | None = None


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _dhash(image: Image.Image, hash_size: int = 8) -> int:
    gray = image.convert("L").resize(
        (hash_size + 1, hash_size),
        resample=Image.Resampling.LANCZOS,
    )
    pixels = np.asarray(gray, dtype=np.int16)
    differences = pixels[:, :-1] > pixels[:, 1:]
    value = 0
    for bit in differences.ravel():
        value = (value << 1) | int(bit)
    return value


def canonical_rotation_dhash(image: Image.Image) -> int:
    """Return the minimum dHash over right-angle rotations.

    Pore patches have no medically meaningful upright orientation. Canonical
    rotation hashing prevents rotated copies from crossing data splits.
    """

    hashes = []
    current = image
    for _ in range(4):
        hashes.append(_dhash(current))
        current = current.transpose(Image.Transpose.ROTATE_90)
    return min(hashes)


def inspect_image(path: Path, root: Path, label: int) -> ImageRecord:
    record = ImageRecord(
        path=path.relative_to(root).as_posix(),
        label=label,
        label_name=CLASS_NAMES[label],
    )
    try:
        payload = path.read_bytes()
        with Image.open(path) as image:
            image.load()
            record.width, record.height = image.size
            record.mode = image.mode
            record.sha256 = hashlib.sha256(payload).hexdigest()
            record.dhash = f"{canonical_rotation_dhash(image):016x}"
            record.status = "inspected"
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        record.status = "corrupt"
        record.error = f"{type(exc).__name__}: {exc}"
    return record


def discover_images(dataset_root: Path) -> list[tuple[Path, int]]:
    discovered: list[tuple[Path, int]] = []
    missing = []
    for label in CLASS_NAMES:
        class_dir = dataset_root / str(label)
        if not class_dir.is_dir():
            missing.append(str(class_dir))
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                discovered.append((path, label))
    if missing:
        raise FileNotFoundError(
            "Expected class directories 1 through 5. Missing: " + ", ".join(missing)
        )
    if not discovered:
        raise FileNotFoundError(f"No supported images found under {dataset_root}")
    return discovered


def cluster_hashes(records: list[ImageRecord], max_distance: int) -> None:
    """Assign exact duplicate groups and conservative split-only groups.

    Identical canonical dHashes are candidates for deduplication. Distinct
    hashes within ``max_distance`` are retained but locked to the same split.
    This separates conservative leakage prevention from destructive cleaning.
    """

    valid = [record for record in records if record.status == "inspected"]
    unique_hashes = sorted({int(record.dhash, 16) for record in valid if record.dhash})
    union_find = UnionFind(len(unique_hashes))

    for left_index, left_hash in enumerate(unique_hashes):
        for right_index in range(left_index + 1, len(unique_hashes)):
            if (left_hash ^ unique_hashes[right_index]).bit_count() <= max_distance:
                union_find.union(left_index, right_index)

    root_to_hashes: dict[int, list[int]] = defaultdict(list)
    for index, hash_value in enumerate(unique_hashes):
        root_to_hashes[union_find.find(index)].append(hash_value)

    hash_to_split_group: dict[int, str] = {}
    ordered_groups = sorted(
        root_to_hashes.values(),
        key=lambda values: min(values),
    )
    for group_number, hash_values in enumerate(ordered_groups, start=1):
        group_id = f"sg{group_number:05d}"
        for hash_value in hash_values:
            hash_to_split_group[hash_value] = group_id

    hash_to_duplicate_group = {
        hash_value: f"dg{index:05d}"
        for index, hash_value in enumerate(unique_hashes, start=1)
    }

    for record in valid:
        hash_value = int(record.dhash, 16)
        record.duplicate_group_id = hash_to_duplicate_group[hash_value]
        record.split_group_id = hash_to_split_group[hash_value]


def resolve_groups(records: list[ImageRecord]) -> None:
    """Exclude conflicts and retain one representative per perceptual group."""

    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        if record.status == "inspected" and record.duplicate_group_id:
            grouped[record.duplicate_group_id].append(record)

    for group_records in grouped.values():
        labels = {record.label for record in group_records}
        if len(labels) > 1:
            for record in group_records:
                record.status = "label_conflict"
            continue

        representatives = sorted(
            group_records,
            key=lambda record: (
                -((record.width or 0) * (record.height or 0)),
                record.path,
            ),
        )
        representatives[0].status = "keep"
        for duplicate in representatives[1:]:
            duplicate.status = "duplicate"


def stratified_split(
    records: list[ImageRecord],
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> None:
    test_fraction = 1.0 - train_fraction - validation_fraction
    if min(train_fraction, validation_fraction, test_fraction) <= 0:
        raise ValueError("Train, validation, and test fractions must all be positive")

    retained = [record for record in records if record.status == "keep"]
    split_groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in retained:
        if not record.split_group_id:
            raise ValueError(f"Missing split group for {record.path}")
        split_groups[record.split_group_id].append(record)

    group_ids = sorted(split_groups)
    # Cross-label near-neighbour groups are rare. Their modal label is used only
    # for approximate stratification; every member still receives one split.
    group_labels = [
        Counter(record.label for record in split_groups[group_id]).most_common(1)[0][0]
        for group_id in group_ids
    ]
    temporary_fraction = validation_fraction + test_fraction
    train_groups, temporary_groups = _safe_train_test_split(
        group_ids,
        group_labels,
        test_size=temporary_fraction,
        seed=seed,
    )
    temporary_labels = [
        Counter(record.label for record in split_groups[group_id]).most_common(1)[0][0]
        for group_id in temporary_groups
    ]
    relative_test_fraction = test_fraction / temporary_fraction
    validation_groups, test_groups = _safe_train_test_split(
        temporary_groups,
        temporary_labels,
        test_size=relative_test_fraction,
        seed=seed,
    )

    assignments = {
        **{group_id: "train" for group_id in train_groups},
        **{group_id: "validation" for group_id in validation_groups},
        **{group_id: "test" for group_id in test_groups},
    }
    for record in retained:
        record.split = assignments[record.split_group_id]


def _safe_train_test_split(
    items: list[str],
    labels: list[int],
    test_size: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Use stratification when the group counts make it mathematically valid."""

    label_counts = Counter(labels)
    class_count = len(label_counts)
    test_count = math.ceil(len(items) * test_size)
    train_count = len(items) - test_count
    can_stratify = (
        min(label_counts.values(), default=0) >= 2
        and test_count >= class_count
        and train_count >= class_count
    )
    train_items, test_items = train_test_split(
        items,
        test_size=test_size,
        random_state=seed,
        stratify=labels if can_stratify else None,
    )
    return list(train_items), list(test_items)


def _write_csv(path: Path, records: Iterable[ImageRecord]) -> None:
    fieldnames = list(ImageRecord.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def build_summary(
    records: list[ImageRecord],
    dataset_root: Path,
    max_distance: int,
    seed: int,
) -> dict:
    retained = [record for record in records if record.status == "keep"]
    split_counts = {
        split: dict(
            sorted(
                Counter(
                    record.label for record in retained if record.split == split
                ).items()
            )
        )
        for split in ("train", "validation", "test")
    }
    dimensions = Counter(
        f"{record.width}x{record.height}"
        for record in records
        if record.width and record.height
    )
    return {
        "dataset_root": str(dataset_root.resolve()),
        "class_names": CLASS_NAMES,
        "total_files": len(records),
        "status_counts": dict(sorted(Counter(r.status for r in records).items())),
        "original_class_counts": dict(
            sorted(Counter(record.label for record in records).items())
        ),
        "retained_class_counts": dict(
            sorted(Counter(record.label for record in retained).items())
        ),
        "split_class_counts": split_counts,
        "image_dimensions": dict(dimensions.most_common()),
        "perceptual_hash": {
            "algorithm": "rotation-canonical 64-bit dHash",
            "deduplication_distance": 0,
            "maximum_split_group_hamming_distance": max_distance,
        },
        "split_seed": seed,
    }


def prepare_dataset(
    dataset_root: Path,
    output_dir: Path,
    max_dhash_distance: int = 2,
    seed: int = 42,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict:
    dataset_root = dataset_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = [
        inspect_image(path, dataset_root, label)
        for path, label in discover_images(dataset_root)
    ]
    cluster_hashes(records, max_distance=max_dhash_distance)
    resolve_groups(records)
    stratified_split(
        records,
        seed=seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )

    records.sort(key=lambda record: (record.label, record.path))
    retained = [record for record in records if record.status == "keep"]
    summary = build_summary(
        records,
        dataset_root=dataset_root,
        max_distance=max_dhash_distance,
        seed=seed,
    )

    _write_csv(output_dir / "audit.csv", records)
    _write_csv(output_dir / "manifest.csv", retained)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/pore_data_set_224"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/data"))
    parser.add_argument(
        "--max-dhash-distance",
        type=int,
        default=2,
        choices=range(0, 9),
        metavar="[0-8]",
        help=(
            "Near-neighbour split-group threshold; only exact canonical hashes "
            "are deduplicated (default: 2 of 64 bits)"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_dataset(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        max_dhash_distance=args.max_dhash_distance,
        seed=args.seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
