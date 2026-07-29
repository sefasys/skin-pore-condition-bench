"""Evaluate a saved model and generate publication-ready diagnostic figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from pore_assessment.metrics import CLASS_NAMES, classification_metrics
from pore_assessment.preprocessing import PREPROCESSING_METHODS
from pore_assessment.preprocessing_cache import cache_image_root, validate_cache
from pore_assessment.train import (
    MODEL_ARCHITECTURES,
    PoreDataset,
    build_model,
    build_transforms,
)

COLORS = {
    "train": "#2A6FBB",
    "validation": "#D97706",
    "test": "#6B7280",
    "accent": "#7C3AED",
}


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_dataset_distribution(frame: pd.DataFrame, path: Path) -> None:
    counts = (
        frame.groupby(["split", "label"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=["train", "validation", "test"], fill_value=0)
        .reindex(columns=range(1, 6), fill_value=0)
    )
    figure, axis = plt.subplots(figsize=(9, 5))
    x_positions = np.arange(len(CLASS_NAMES))
    width = 0.24
    for index, split in enumerate(counts.index):
        values = counts.loc[split].to_numpy()
        bars = axis.bar(
            x_positions + (index - 1) * width,
            values,
            width,
            label=split.title(),
            color=COLORS[split],
        )
        axis.bar_label(bars, fontsize=8, padding=2)
    axis.set_title("Retained dataset distribution")
    axis.set_xlabel("Ordered pore-condition class")
    axis.set_ylabel("Images")
    axis.set_xticks(
        x_positions, [name.replace("_", " ").title() for name in CLASS_NAMES]
    )
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    _save_figure(figure, path)


def plot_learning_curves(results: dict, path: Path) -> None:
    history = results.get("history", [])
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].plot(
        epochs,
        [row["train_loss"] for row in history],
        marker="o",
        label="Train",
        color=COLORS["train"],
    )
    axes[0, 0].plot(
        epochs,
        [row["validation_loss"] for row in history],
        marker="o",
        label="Validation",
        color=COLORS["validation"],
    )
    axes[0, 0].set_title("Cross-entropy loss")

    for split, color in (
        ("train", COLORS["train"]),
        ("validation", COLORS["validation"]),
    ):
        axes[0, 1].plot(
            epochs,
            [row[split]["macro_f1"] for row in history],
            marker="o",
            label=split.title(),
            color=color,
        )
    axes[0, 1].set_title("Macro F1")
    axes[0, 1].set_ylim(0, 1)

    for split, color in (
        ("train", COLORS["train"]),
        ("validation", COLORS["validation"]),
    ):
        axes[1, 0].plot(
            epochs,
            [row[split]["quadratic_weighted_kappa"] for row in history],
            marker="o",
            label=split.title(),
            color=color,
        )
    axes[1, 0].set_title("Quadratic weighted kappa")
    axes[1, 0].set_ylim(-0.1, 1)

    for split, color in (
        ("train", COLORS["train"]),
        ("validation", COLORS["validation"]),
    ):
        axes[1, 1].plot(
            epochs,
            [row[split]["mean_absolute_class_error"] for row in history],
            marker="o",
            label=split.title(),
            color=color,
        )
    axes[1, 1].set_title("Mean absolute class error (lower is better)")

    for axis in axes.ravel():
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xticks(epochs)
    figure.suptitle("Training and validation learning curves", fontsize=14)
    _save_figure(figure, path)


def _annotate_matrix(axis: plt.Axes, matrix: np.ndarray, decimal: bool) -> None:
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            label = f"{value:.2f}" if decimal else str(int(value))
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=9,
            )


def plot_confusion_matrices(metrics: dict, path: Path) -> None:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_totals != 0,
    )
    labels = [name.replace("_", " ").title() for name in CLASS_NAMES]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, values, title, decimal in (
        (axes[0], matrix, "Counts", False),
        (axes[1], normalized, "Row-normalized recall", True),
    ):
        image = axis.imshow(values, cmap="Blues", vmin=0)
        _annotate_matrix(axis, values, decimal=decimal)
        axis.set_title(title)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_xticks(range(5), labels, rotation=35, ha="right")
        axis.set_yticks(range(5), labels)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("Pore-condition confusion matrices", fontsize=14)
    _save_figure(figure, path)


def plot_per_class_metrics(metrics: dict, path: Path) -> None:
    class_metrics = metrics["per_class"]
    x_positions = np.arange(len(CLASS_NAMES))
    width = 0.24
    figure, axis = plt.subplots(figsize=(10, 5))
    for offset, metric, color in (
        (-1, "precision", "#2A6FBB"),
        (0, "recall", "#D97706"),
        (1, "f1", "#7C3AED"),
    ):
        values = [class_metrics[name][metric] for name in CLASS_NAMES]
        axis.bar(
            x_positions + offset * width,
            values,
            width,
            label=metric.title(),
            color=color,
        )
    axis.set_title("Per-class performance")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1)
    axis.set_xticks(
        x_positions,
        [name.replace("_", " ").title() for name in CLASS_NAMES],
    )
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    _save_figure(figure, path)


def plot_ordinal_error_distances(
    targets: np.ndarray,
    predictions: np.ndarray,
    path: Path,
) -> None:
    distances = np.abs(targets - predictions)
    counts = np.bincount(distances, minlength=5)
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(range(5), counts, color=COLORS["accent"])
    axis.bar_label(bars, padding=3)
    axis.set_title("Ordinal prediction-error distance")
    axis.set_xlabel("Absolute distance between true and predicted class")
    axis.set_ylabel("Images")
    axis.set_xticks(range(5))
    axis.spines[["top", "right"]].set_visible(False)
    _save_figure(figure, path)


def plot_calibration(
    targets: np.ndarray,
    probabilities: np.ndarray,
    path: Path,
    bins: int = 10,
) -> None:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == targets
    boundaries = np.linspace(0, 1, bins + 1)
    centers = (boundaries[:-1] + boundaries[1:]) / 2
    accuracies = np.full(bins, np.nan)
    mean_confidences = np.full(bins, np.nan)
    counts = np.zeros(bins, dtype=int)

    for index in range(bins):
        if index == 0:
            mask = (confidence >= boundaries[index]) & (
                confidence <= boundaries[index + 1]
            )
        else:
            mask = (confidence > boundaries[index]) & (
                confidence <= boundaries[index + 1]
            )
        counts[index] = int(mask.sum())
        if counts[index]:
            accuracies[index] = correct[mask].mean()
            mean_confidences[index] = confidence[mask].mean()

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(7, 7),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#6B7280", label="Ideal")
    valid = ~np.isnan(accuracies)
    axes[0].plot(
        mean_confidences[valid],
        accuracies[valid],
        marker="o",
        color=COLORS["accent"],
        label="Model",
    )
    axes[0].set_title("Confidence calibration")
    axes[0].set_xlabel("Mean confidence")
    axes[0].set_ylabel("Observed accuracy")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    axes[1].bar(centers, counts, width=0.09, color="#9CA3AF")
    axes[1].set_xlabel("Prediction confidence")
    axes[1].set_ylabel("Images")
    axes[1].set_xlim(0, 1)
    axes[1].spines[["top", "right"]].set_visible(False)
    _save_figure(figure, path)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    targets = []
    probabilities = []
    total_loss = 0.0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            total_loss += float(criterion(logits, labels).item()) * labels.size(0)
            targets.extend(labels.cpu().tolist())
            probabilities.extend(torch.softmax(logits, dim=1).cpu().tolist())
    return (
        total_loss / len(targets),
        np.asarray(targets, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
    )


def generate_report(args: argparse.Namespace) -> dict:
    if args.split == "test" and not args.allow_test:
        raise ValueError(
            "Test evaluation is locked. Add --allow-test only after model selection."
        )

    frame = pd.read_csv(args.manifest)
    evaluation_frame = frame[frame["split"] == args.split].copy().reset_index(drop=True)
    if evaluation_frame.empty:
        raise ValueError(f"No rows found for split: {args.split}")

    dataset_root = args.dataset_root.resolve()
    device = torch.device(
        args.device
        if args.device != "auto"
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    checkpoint_architecture = checkpoint.get("architecture", "resnet18")
    architecture = (
        checkpoint_architecture if args.architecture == "auto" else args.architecture
    )
    if args.architecture != "auto" and args.architecture != checkpoint_architecture:
        raise ValueError(
            "Requested architecture does not match the checkpoint: "
            f"{args.architecture!r} != {checkpoint_architecture!r}"
        )
    checkpoint_preprocessing = checkpoint.get("preprocessing")
    preprocessing = (
        checkpoint_preprocessing
        if args.preprocessing == "auto" and checkpoint_preprocessing
        else "raw"
        if args.preprocessing == "auto"
        else args.preprocessing
    )
    if (
        checkpoint_preprocessing
        and args.preprocessing != "auto"
        and args.preprocessing != checkpoint_preprocessing
    ):
        raise ValueError(
            "Requested preprocessing does not match the checkpoint: "
            f"{args.preprocessing!r} != {checkpoint_preprocessing!r}"
        )
    cache_root = (
        args.preprocessing_cache_root.resolve()
        if args.preprocessing_cache_root
        else None
    )
    cache_metadata = None
    input_root = dataset_root
    cached_png = False
    transform_preprocessing = preprocessing
    if cache_root:
        if preprocessing == "raw":
            raise ValueError(
                "--preprocessing-cache-root requires non-raw preprocessing"
            )
        cache_metadata = validate_cache(
            cache_root,
            preprocessing,
            args.manifest,
            args.image_size,
        )
        input_root = cache_image_root(cache_root, preprocessing)
        cached_png = True
        transform_preprocessing = "raw"
    _, evaluation_transform = build_transforms(
        args.image_size,
        transform_preprocessing,
    )
    loader = DataLoader(
        PoreDataset(
            evaluation_frame,
            input_root,
            evaluation_transform,
            cached_png=cached_png,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        architecture=architecture,
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    loss, targets, probabilities = evaluate(model, loader, device)
    predictions = probabilities.argmax(axis=1)
    metrics = {
        "split": args.split,
        "architecture": architecture,
        "preprocessing": preprocessing,
        "preprocessing_cache_lossless_verified": (
            cache_metadata["lossless_png_verified"] if cache_metadata else None
        ),
        "loss": loss,
        "images": len(targets),
        **classification_metrics(targets, predictions, probabilities),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame = evaluation_frame[["path", "label", "label_name"]].copy()
    prediction_frame["predicted_label"] = predictions + 1
    prediction_frame["predicted_label_name"] = [
        CLASS_NAMES[prediction] for prediction in predictions
    ]
    prediction_frame["confidence"] = probabilities.max(axis=1)
    prediction_frame["absolute_class_error"] = np.abs(targets - predictions)
    for index, class_name in enumerate(CLASS_NAMES):
        prediction_frame[f"probability_{class_name}"] = probabilities[:, index]
    prediction_frame.to_csv(args.output_dir / "predictions.csv", index=False)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")

    plot_dataset_distribution(frame, args.output_dir / "dataset_distribution.png")
    plot_confusion_matrices(metrics, args.output_dir / "confusion_matrices.png")
    plot_per_class_metrics(metrics, args.output_dir / "per_class_metrics.png")
    plot_ordinal_error_distances(
        targets,
        predictions,
        args.output_dir / "ordinal_error_distance.png",
    )
    plot_calibration(
        targets,
        probabilities,
        args.output_dir / "calibration.png",
    )
    run_results = args.checkpoint.parent / "results.json"
    if run_results.is_file():
        with run_results.open(encoding="utf-8") as handle:
            plot_learning_curves(
                json.load(handle),
                args.output_dir / "learning_curves.png",
            )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/data/manifest.csv")
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/pore_data_set_224"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/raw-resnet18/best.pt"),
    )
    parser.add_argument(
        "--architecture",
        choices=("auto", *MODEL_ARCHITECTURES),
        default="auto",
        help="Infer from checkpoint metadata by default",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/report"))
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="validation",
    )
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--preprocessing",
        choices=("auto", *PREPROCESSING_METHODS),
        default="auto",
        help="Infer from checkpoint metadata by default; legacy checkpoints use raw",
    )
    parser.add_argument(
        "--preprocessing-cache-root",
        type=Path,
        help="Use the lossless deterministic cache used during training",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Data-loader subprocesses (default: 0 for restricted environments)",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    metrics = generate_report(parse_args())
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
