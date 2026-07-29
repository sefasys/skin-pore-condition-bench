"""Train and evaluate the first raw-image pore-condition baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    convnext_tiny,
    efficientnet_b0,
    resnet18,
)

from pore_assessment.metrics import CLASS_NAMES, classification_metrics
from pore_assessment.preprocessing import (
    DEFAULT_CONFIG,
    PREPROCESSING_METHODS,
    DeterministicPreprocessing,
)
from pore_assessment.preprocessing_cache import (
    cache_image_root,
    validate_cache,
)

MODEL_ARCHITECTURES = ("resnet18", "efficientnet_b0", "convnext_tiny")


class PoreDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        dataset_root: Path,
        transform: transforms.Compose,
        cached_png: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.dataset_root = dataset_root
        self.transform = transform
        self.cached_png = cached_png

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        relative_path = Path(row["path"])
        if self.cached_png:
            relative_path = relative_path.with_suffix(".png")
        path = self.dataset_root / relative_path
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, int(row["label"]) - 1


class RandomRightAngleRotation:
    """Rotate without interpolation or artificial triangular borders."""

    _TRANSPOSES = (
        None,
        Image.Transpose.ROTATE_90,
        Image.Transpose.ROTATE_180,
        Image.Transpose.ROTATE_270,
    )

    def __call__(self, image: Image.Image) -> Image.Image:
        operation = random.choice(self._TRANSPOSES)
        return image if operation is None else image.transpose(operation)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(
    image_size: int,
    preprocessing: str = "raw",
) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    deterministic_preprocessing = DeterministicPreprocessing(preprocessing)
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            deterministic_preprocessing,
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            RandomRightAngleRotation(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            deterministic_preprocessing,
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, evaluation_transform


def build_model(
    architecture: str = "resnet18",
    pretrained: bool = False,
    weights_path: Path | None = None,
) -> nn.Module:
    if architecture not in MODEL_ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture {architecture!r}; choose from {MODEL_ARCHITECTURES}"
        )
    builders = {
        "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
        "efficientnet_b0": (efficientnet_b0, EfficientNet_B0_Weights.DEFAULT),
        "convnext_tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT),
    }
    builder, default_weights = builders[architecture]
    if weights_path is not None:
        model = builder(weights=None)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    else:
        model = builder(weights=default_weights if pretrained else None)

    if architecture == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    else:
        final_layer = model.classifier[-1]
        model.classifier[-1] = nn.Linear(
            final_layer.in_features,
            len(CLASS_NAMES),
        )
    return model


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def class_weights(
    frame: pd.DataFrame,
    weighting: str,
) -> torch.Tensor | None:
    if weighting not in {"balanced", "sqrt_balanced", "none"}:
        raise ValueError(f"Unknown class weighting: {weighting}")
    if weighting == "none":
        return None
    counts = frame["label"].value_counts().reindex(range(1, 6), fill_value=0)
    if (counts == 0).any():
        missing = counts[counts == 0].index.tolist()
        raise ValueError(f"Training split is missing labels: {missing}")
    weights = len(frame) / (len(CLASS_NAMES) * counts.to_numpy(dtype=np.float64))
    if weighting == "sqrt_balanced":
        weights = np.sqrt(weights)
    return torch.tensor(weights, dtype=torch.float32)


def classifier_parameter_ids(
    model: nn.Module,
    architecture: str,
) -> set[int]:
    classifier = model.fc if architecture == "resnet18" else model.classifier
    return {id(parameter) for parameter in classifier.parameters()}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> tuple[float, dict]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device)
            labels = labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item()) * labels.size(0)
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            probabilities.extend(torch.softmax(logits, dim=1).detach().cpu().tolist())

    if not targets:
        raise RuntimeError("No batches were processed")
    return total_loss / len(targets), classification_metrics(
        targets,
        predictions,
        probabilities,
    )


def train(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0, 1)")
    if args.last_checkpoint_interval < 1:
        raise ValueError("--last-checkpoint-interval must be positive")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.backbone_learning_rate is not None and args.backbone_learning_rate <= 0:
        raise ValueError("--backbone-learning-rate must be positive")
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    frame = pd.read_csv(args.manifest)
    required_columns = {"path", "label", "split"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")

    dataset_root = (
        Path(args.dataset_root) if args.dataset_root else Path("data/pore_data_set_224")
    ).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset root not found: {dataset_root}. Pass --dataset-root explicitly."
        )

    device = torch.device(
        args.device
        if args.device != "auto"
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    cache_root = (
        args.preprocessing_cache_root.resolve()
        if args.preprocessing_cache_root
        else None
    )
    cache_metadata = None
    input_root = dataset_root
    cached_png = False
    transform_preprocessing = args.preprocessing
    if cache_root:
        if args.preprocessing == "raw":
            raise ValueError(
                "--preprocessing-cache-root requires non-raw preprocessing"
            )
        cache_metadata = validate_cache(
            cache_root,
            args.preprocessing,
            args.manifest,
            args.image_size,
        )
        input_root = cache_image_root(cache_root, args.preprocessing)
        cached_png = True
        transform_preprocessing = "raw"

    train_transform, evaluation_transform = build_transforms(
        args.image_size,
        transform_preprocessing,
    )
    split_frames = {
        split: frame[frame["split"] == split].copy()
        for split in ("train", "validation", "test")
    }
    if split_frames["train"].empty or split_frames["validation"].empty:
        raise ValueError("Manifest must contain non-empty train and validation splits")

    train_loader = DataLoader(
        PoreDataset(
            split_frames["train"],
            input_root,
            train_transform,
            cached_png=cached_png,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        PoreDataset(
            split_frames["validation"],
            input_root,
            evaluation_transform,
            cached_png=cached_png,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    weights_path = args.weights_path.resolve() if args.weights_path else None
    if weights_path and not weights_path.is_file():
        raise FileNotFoundError(f"Pretrained weights not found: {weights_path}")
    model = build_model(
        architecture=args.architecture,
        pretrained=args.pretrained,
        weights_path=weights_path,
    ).to(device)
    criterion_weights = class_weights(
        split_frames["train"],
        args.class_weighting,
    )
    if criterion_weights is not None:
        criterion_weights = criterion_weights.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=criterion_weights,
        label_smoothing=args.label_smoothing,
    )
    if args.backbone_learning_rate is None:
        optimizer_parameters = model.parameters()
    else:
        head_parameter_ids = classifier_parameter_ids(model, args.architecture)
        backbone_parameters = [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in head_parameter_ids
        ]
        head_parameters = [
            parameter
            for parameter in model.parameters()
            if id(parameter) in head_parameter_ids
        ]
        optimizer_parameters = [
            {
                "params": backbone_parameters,
                "lr": args.backbone_learning_rate,
                "name": "backbone",
            },
            {
                "params": head_parameters,
                "lr": args.learning_rate,
                "name": "classifier",
            },
        ]
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.minimum_learning_rate,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_score = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    first_epoch = 1
    started = time.time()

    if args.resume:
        last_checkpoint_path = args.output_dir / "last.pt"
        results_path = args.output_dir / "results.json"
        if not last_checkpoint_path.is_file() or not results_path.is_file():
            raise FileNotFoundError(
                "--resume requires last.pt and results.json in the output directory"
            )
        checkpoint = torch.load(
            last_checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        with results_path.open(encoding="utf-8") as handle:
            previous_results = json.load(handle)
        previous_configuration = previous_results.get("configuration", {})
        expected_resume_configuration = {
            "architecture": args.architecture,
            "preprocessing": args.preprocessing,
            "class_weighting": args.class_weighting,
            "label_smoothing": args.label_smoothing,
            "selection_metric": args.selection_metric,
            "image_size": args.image_size,
            "learning_rate": args.learning_rate,
            "backbone_learning_rate": args.backbone_learning_rate,
            "weight_decay": args.weight_decay,
        }
        legacy_defaults = {
            "architecture": "resnet18",
            "preprocessing": "raw",
            "label_smoothing": 0.0,
            "backbone_learning_rate": None,
        }
        resume_mismatches = {
            key: (
                previous_configuration.get(key, legacy_defaults.get(key)),
                expected,
            )
            for key, expected in expected_resume_configuration.items()
            if previous_configuration.get(key, legacy_defaults.get(key)) != expected
        }
        if resume_mismatches:
            raise ValueError(
                "Cannot resume with a different training configuration: "
                f"{resume_mismatches}"
            )
        history = previous_results["history"]
        best_score = previous_results["best_validation_score"]
        best_epoch = previous_results["best_epoch"]
        first_epoch = int(checkpoint["epoch"]) + 1
        history = [row for row in history if int(row["epoch"]) < first_epoch]
        epochs_without_improvement = max(0, first_epoch - 1 - best_epoch)
        print(f"Resuming at epoch {first_epoch}", flush=True)

    for epoch in range(first_epoch, args.epochs + 1):
        epoch_started = time.time()
        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
        )
        validation_loss, validation_metrics = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
            max_batches=args.max_eval_batches,
        )
        result = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "learning_rates": {
                group.get("name", f"group_{index}"): group["lr"]
                for index, group in enumerate(optimizer.param_groups)
            },
            "elapsed_seconds": time.time() - epoch_started,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(result)
        print(json.dumps(result), flush=True)

        score = validation_metrics[args.selection_metric]
        scheduler.step(score)
        if score > best_score + args.minimum_improvement:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": validation_metrics,
                    "class_names": CLASS_NAMES,
                    "architecture": args.architecture,
                    "selection_metric": args.selection_metric,
                    "selection_score": score,
                    "preprocessing": args.preprocessing,
                    "preprocessing_parameters": DEFAULT_CONFIG.to_dict(),
                },
                args.output_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1

        should_stop = (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        )
        should_save_resume_checkpoint = (
            epoch % args.last_checkpoint_interval == 0
            or epoch == args.epochs
            or should_stop
        )
        if should_save_resume_checkpoint:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": validation_metrics,
                    "class_names": CLASS_NAMES,
                    "architecture": args.architecture,
                    "preprocessing": args.preprocessing,
                    "preprocessing_parameters": DEFAULT_CONFIG.to_dict(),
                },
                args.output_dir / "last.pt",
            )

        partial_result = {
            "configuration": {
                "manifest": str(args.manifest.resolve()),
                "dataset_root": str(dataset_root),
                "architecture": args.architecture,
                "pretrained": args.pretrained or weights_path is not None,
                "weights_path": str(weights_path) if weights_path else None,
                "weights_sha256": sha256_file(weights_path) if weights_path else None,
                "preprocessing": args.preprocessing,
                "preprocessing_parameters": DEFAULT_CONFIG.to_dict(),
                "preprocessing_cache_root": str(cache_root) if cache_root else None,
                "preprocessing_cache_lossless_verified": (
                    cache_metadata["lossless_png_verified"] if cache_metadata else None
                ),
                "class_weighting": args.class_weighting,
                "label_smoothing": args.label_smoothing,
                "selection_metric": args.selection_metric,
                "epochs_requested": args.epochs,
                "batch_size": args.batch_size,
                "image_size": args.image_size,
                "learning_rate": args.learning_rate,
                "backbone_learning_rate": args.backbone_learning_rate,
                "weight_decay": args.weight_decay,
                "last_checkpoint_interval": args.last_checkpoint_interval,
                "seed": args.seed,
                "device": str(device),
                "test_evaluated": False,
                "resumed": args.resume,
            },
            "best_epoch": best_epoch,
            "best_validation_score": best_score,
            "epochs_completed": epoch,
            "stopped_early": False,
            "elapsed_seconds": time.time() - started,
            "history": history,
        }
        with (args.output_dir / "results.json").open("w", encoding="utf-8") as handle:
            json.dump(partial_result, handle, indent=2)
            handle.write("\n")

        if should_stop:
            print(
                f"Early stopping after epoch {epoch}; no "
                f"{args.selection_metric} improvement for "
                f"{args.early_stopping_patience} epochs.",
                flush=True,
            )
            break

    result = {
        "configuration": {
            "manifest": str(args.manifest.resolve()),
            "dataset_root": str(dataset_root),
            "architecture": args.architecture,
            "pretrained": args.pretrained or weights_path is not None,
            "weights_path": str(weights_path) if weights_path else None,
            "weights_sha256": sha256_file(weights_path) if weights_path else None,
            "preprocessing": args.preprocessing,
            "preprocessing_parameters": DEFAULT_CONFIG.to_dict(),
            "preprocessing_cache_root": str(cache_root) if cache_root else None,
            "preprocessing_cache_lossless_verified": (
                cache_metadata["lossless_png_verified"] if cache_metadata else None
            ),
            "class_weighting": args.class_weighting,
            "label_smoothing": args.label_smoothing,
            "selection_metric": args.selection_metric,
            "epochs_requested": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "learning_rate": args.learning_rate,
            "backbone_learning_rate": args.backbone_learning_rate,
            "weight_decay": args.weight_decay,
            "last_checkpoint_interval": args.last_checkpoint_interval,
            "seed": args.seed,
            "device": str(device),
            "test_evaluated": args.evaluate_test,
            "resumed": args.resume,
        },
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "epochs_completed": len(history),
        "stopped_early": len(history) < args.epochs,
        "elapsed_seconds": time.time() - started,
        "history": history,
    }

    if args.evaluate_test:
        checkpoint = torch.load(
            args.output_dir / "best.pt",
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model_state"])
        test_loader = DataLoader(
            PoreDataset(
                split_frames["test"],
                input_root,
                evaluation_transform,
                cached_png=cached_png,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        test_loss, test_metrics = run_epoch(
            model,
            test_loader,
            criterion,
            device,
            max_batches=args.max_eval_batches,
        )
        result["test"] = {"loss": test_loss, **test_metrics}

    with (args.output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/data/manifest.csv")
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/raw-resnet18"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--backbone-learning-rate",
        type=float,
        help=(
            "Optional lower LR for pretrained feature layers; "
            "--learning-rate then applies to the classifier head"
        ),
    )
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Data-loader subprocesses (default: 0 for restricted environments)",
    )
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from last.pt and results.json in the output directory",
    )
    parser.add_argument("--weights-path", type=Path)
    parser.add_argument(
        "--architecture",
        choices=MODEL_ARCHITECTURES,
        default="resnet18",
    )
    parser.add_argument(
        "--preprocessing",
        choices=PREPROCESSING_METHODS,
        default="raw",
        help="Deterministic input method applied after resize (default: raw)",
    )
    parser.add_argument(
        "--preprocessing-cache-root",
        type=Path,
        help=(
            "Use a lossless cache created by preprocessing_cache; the "
            "checkpoint still records the scientific preprocessing method"
        ),
    )
    parser.add_argument(
        "--class-weighting",
        choices=("balanced", "sqrt_balanced", "none"),
        default="balanced",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--selection-metric",
        choices=("quadratic_weighted_kappa", "macro_f1", "exact_accuracy"),
        default="quadratic_weighted_kappa",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=7)
    parser.add_argument("--minimum-improvement", type=float, default=1e-4)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-factor", type=float, default=0.3)
    parser.add_argument(
        "--last-checkpoint-interval",
        type=int,
        default=1,
        help="Save resumable optimizer state every N epochs (default: 1)",
    )
    pretrained = parser.add_mutually_exclusive_group()
    pretrained.add_argument(
        "--pretrained", dest="pretrained", action="store_true", default=True
    )
    pretrained.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train(args)
    print(
        f"Best validation {result['configuration']['selection_metric']}: "
        f"{result['best_validation_score']:.4f} "
        f"(epoch {result['best_epoch']})"
    )


if __name__ == "__main__":
    main()
