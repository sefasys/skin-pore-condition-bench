"""Orchestrate the reproducible pore-condition preprocessing study."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

STAGES = ("prepare", "audit", "cache", "train", "report", "summarize")


def _command(module: str, *arguments: object) -> list[str]:
    return [sys.executable, "-m", module, *(str(value) for value in arguments)]


def _run(command: list[str], dry_run: bool) -> None:
    print("+ " + shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def _resolve(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _training_complete(run_dir: Path) -> bool:
    results_path = run_dir / "results.json"
    checkpoint_path = run_dir / "best.pt"
    if not results_path.is_file() or not checkpoint_path.is_file():
        return False
    try:
        with results_path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        completed = int(result["epochs_completed"])
        requested = int(result["configuration"]["epochs_requested"])
        stopped_early = bool(result.get("stopped_early", False))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return stopped_early or completed >= requested


def run_pipeline(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    project_root = args.project_root.resolve()
    stages = set(args.stages)

    dataset_root = _resolve(config["dataset_root"], project_root)
    artifact_root = _resolve(config["artifact_root"], project_root)
    run_root = _resolve(config["run_root"], project_root)
    summary_dir = _resolve(config["summary_dir"], project_root)
    data_dir = artifact_root / "data"
    manifest = data_dir / "manifest.csv"
    cache_root = artifact_root / "preprocessing-cache"
    methods = list(config["methods"])
    enhanced_methods = [method for method in methods if method != "raw"]
    seeds = [int(seed) for seed in config["seeds"]]

    if "prepare" in stages:
        command = _command(
            "pore_assessment.prepare",
            "--dataset-root",
            dataset_root,
            "--output-dir",
            data_dir,
            "--max-dhash-distance",
            config["max_dhash_distance"],
            "--seed",
            config["split_seed"],
            "--train-fraction",
            config["train_fraction"],
            "--validation-fraction",
            config["validation_fraction"],
        )
        if args.force or not manifest.is_file():
            _run(command, args.dry_run)
        else:
            print(f"SKIP prepare: {manifest} already exists", flush=True)

    if "audit" in stages:
        audit_dir = artifact_root / "preprocessing-audit"
        command = _command(
            "pore_assessment.preprocessing_audit",
            "--manifest",
            manifest,
            "--dataset-root",
            dataset_root,
            "--output-dir",
            audit_dir,
            "--split",
            "validation",
            "--image-size",
            config["image_size"],
        )
        if args.force or not (audit_dir / "summary.csv").is_file():
            _run(command, args.dry_run)
        else:
            print(f"SKIP audit: {audit_dir / 'summary.csv'} exists", flush=True)

    if "cache" in stages and enhanced_methods:
        command = _command(
            "pore_assessment.preprocessing_cache",
            "--manifest",
            manifest,
            "--dataset-root",
            dataset_root,
            "--cache-root",
            cache_root,
            "--methods",
            *enhanced_methods,
            "--image-size",
            config["image_size"],
        )
        _run(command, args.dry_run)

    for method in methods:
        for seed in seeds:
            run_dir = run_root / method / f"seed{seed}"
            if "train" in stages:
                train_command = _command(
                    "pore_assessment.train",
                    "--manifest",
                    manifest,
                    "--dataset-root",
                    dataset_root,
                    "--output-dir",
                    run_dir,
                    "--architecture",
                    config["architecture"],
                    "--preprocessing",
                    method,
                    "--epochs",
                    config["epochs"],
                    "--batch-size",
                    config["batch_size"],
                    "--image-size",
                    config["image_size"],
                    "--learning-rate",
                    config["head_learning_rate"],
                    "--backbone-learning-rate",
                    config["backbone_learning_rate"],
                    "--minimum-learning-rate",
                    config["minimum_learning_rate"],
                    "--weight-decay",
                    config["weight_decay"],
                    "--class-weighting",
                    config["class_weighting"],
                    "--label-smoothing",
                    config["label_smoothing"],
                    "--selection-metric",
                    config["selection_metric"],
                    "--early-stopping-patience",
                    config["early_stopping_patience"],
                    "--minimum-improvement",
                    config["minimum_improvement"],
                    "--scheduler-patience",
                    config["scheduler_patience"],
                    "--scheduler-factor",
                    config["scheduler_factor"],
                    "--workers",
                    config["workers"],
                    "--device",
                    config["device"],
                    "--seed",
                    seed,
                    "--last-checkpoint-interval",
                    1,
                )
                if config.get("pretrained", True):
                    train_command.append("--pretrained")
                else:
                    train_command.append("--no-pretrained")
                if method != "raw":
                    train_command.extend(
                        ["--preprocessing-cache-root", str(cache_root)]
                    )
                for key, flag in (
                    ("max_train_batches", "--max-train-batches"),
                    ("max_eval_batches", "--max-eval-batches"),
                ):
                    if config.get(key) is not None:
                        train_command.extend([flag, str(config[key])])
                if (
                    args.resume
                    and not args.force
                    and (run_dir / "last.pt").is_file()
                    and (run_dir / "results.json").is_file()
                ):
                    train_command.append("--resume")
                if args.force or not _training_complete(run_dir):
                    _run(train_command, args.dry_run)
                else:
                    print(f"SKIP train: {run_dir} is complete", flush=True)

            if "report" in stages:
                split = "test" if args.evaluate_test else "validation"
                report_dir = run_dir / f"{split}-report"
                report_command = _command(
                    "pore_assessment.report",
                    "--manifest",
                    manifest,
                    "--dataset-root",
                    dataset_root,
                    "--checkpoint",
                    run_dir / "best.pt",
                    "--output-dir",
                    report_dir,
                    "--split",
                    split,
                    "--batch-size",
                    config["batch_size"],
                    "--image-size",
                    config["image_size"],
                    "--workers",
                    config["workers"],
                    "--device",
                    config["device"],
                )
                if method != "raw":
                    report_command.extend(
                        ["--preprocessing-cache-root", str(cache_root)]
                    )
                if args.evaluate_test:
                    report_command.append("--allow-test")
                if args.force or not (report_dir / "metrics.json").is_file():
                    _run(report_command, args.dry_run)
                else:
                    print(
                        f"SKIP report: {report_dir / 'metrics.json'} exists",
                        flush=True,
                    )

    if "summarize" in stages:
        split = "test" if args.evaluate_test else "validation"
        target = summary_dir if split == "validation" else summary_dir.parent / "test"
        command = _command(
            "pore_assessment.study_summary",
            "--run-root",
            run_root,
            "--manifest",
            manifest,
            "--output-dir",
            target,
            "--report-split",
            split,
            "--methods",
            *methods,
            "--seeds",
            *seeds,
            "--bootstrap-samples",
            config["bootstrap_samples"],
        )
        _run(command, args.dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/final_protocol.json"),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Resolve relative paths from this directory (default: current directory)",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGES,
        default=list(STAGES),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Explicitly unlock the held-out test after the protocol is frozen",
    )
    return parser.parse_args()


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
