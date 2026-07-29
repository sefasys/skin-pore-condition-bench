"""Aggregate multi-seed reports and compare preprocessing methods with raw RGB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import binomtest

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from pore_assessment.metrics import CLASS_NAMES

METRIC_PATHS = {
    "exact_accuracy": ("exact_accuracy",),
    "macro_f1": ("macro_f1",),
    "balanced_accuracy": ("balanced_accuracy",),
    "quadratic_weighted_kappa": ("quadratic_weighted_kappa",),
    "mean_absolute_class_error": ("mean_absolute_class_error",),
    "two_or_more_class_error_rate": ("two_or_more_class_error_rate",),
    "ece": ("calibration", "expected_calibration_error_10_bins"),
    "macro_roc_auc": ("discrimination", "macro_ovr_roc_auc"),
    "macro_average_precision": ("discrimination", "macro_average_precision"),
}


def _nested(mapping: dict, path: tuple[str, ...]) -> float:
    value: object = mapping
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return float("nan")
        value = value[key]
    return float(value)


def _save(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def load_reports(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict] = []
    class_rows: list[dict] = []
    for method in args.methods:
        for seed in args.seeds:
            report_dir = (
                args.run_root / method / f"seed{seed}" / f"{args.report_split}-report"
            )
            metrics_path = report_dir / "metrics.json"
            if not metrics_path.is_file():
                raise FileNotFoundError(f"Missing report: {metrics_path}")
            with metrics_path.open(encoding="utf-8") as handle:
                metrics = json.load(handle)
            metric_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    **{
                        name: _nested(metrics, path)
                        for name, path in METRIC_PATHS.items()
                    },
                }
            )
            for class_name in CLASS_NAMES:
                values = metrics["per_class"][class_name]
                class_rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "class_name": class_name,
                        "precision": values["precision"],
                        "recall": values["recall"],
                        "f1": values["f1"],
                        "support": values["support"],
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(class_rows)


def summarize_methods(per_seed: pd.DataFrame) -> pd.DataFrame:
    metric_columns = list(METRIC_PATHS)
    grouped = per_seed.groupby("method", sort=False)[metric_columns]
    sections = []
    for statistic in ("mean", "std", "min", "max"):
        suffix = {
            "mean": "mean",
            "std": "std",
            "min": "minimum",
            "max": "maximum",
        }[statistic]
        frame = getattr(grouped, statistic)().rename(
            columns=lambda column, suffix=suffix: f"{column}_{suffix}"
        )
        sections.append(frame)
    summary = pd.concat(sections, axis=1)
    summary["seeds_completed"] = grouped.size()
    return summary.reset_index()


def summarize_classes(per_class: pd.DataFrame) -> pd.DataFrame:
    return (
        per_class.groupby(["method", "class_name"], sort=False)[
            ["precision", "recall", "f1"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )


def load_predictions(
    run_root: Path,
    method: str,
    seed: int,
    report_split: str,
) -> pd.DataFrame:
    path = (
        run_root / method / f"seed{seed}" / f"{report_split}-report" / "predictions.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing predictions: {path}")
    frame = pd.read_csv(path)
    required = {"path", "label", "predicted_label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame[list(required)].copy()


def paired_comparisons(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    rows = []
    paired_frames: dict[tuple[str, int], pd.DataFrame] = {}
    for seed in args.seeds:
        raw = load_predictions(args.run_root, "raw", seed, args.report_split)
        raw = raw.rename(columns={"predicted_label": "raw_prediction"})
        for method in args.methods:
            if method == "raw":
                continue
            enhanced = load_predictions(
                args.run_root, method, seed, args.report_split
            ).rename(columns={"predicted_label": "enhanced_prediction"})
            merged = raw.merge(
                enhanced[["path", "label", "enhanced_prediction"]],
                on=["path", "label"],
                validate="one_to_one",
            )
            if len(merged) != len(raw) or len(merged) != len(enhanced):
                raise ValueError(
                    f"Prediction paths do not align for {method}, seed {seed}"
                )
            raw_correct = merged["raw_prediction"] == merged["label"]
            enhanced_correct = merged["enhanced_prediction"] == merged["label"]
            corrected = int((~raw_correct & enhanced_correct).sum())
            broken = int((raw_correct & ~enhanced_correct).sum())
            discordant = corrected + broken
            p_value = (
                float(
                    binomtest(
                        min(corrected, broken),
                        n=discordant,
                        p=0.5,
                        alternative="two-sided",
                    ).pvalue
                )
                if discordant
                else 1.0
            )
            raw_error = (merged["raw_prediction"] - merged["label"]).abs()
            enhanced_error = (merged["enhanced_prediction"] - merged["label"]).abs()
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "delta_exact_vs_raw": float(
                        enhanced_correct.mean() - raw_correct.mean()
                    ),
                    "delta_qwk_vs_raw": _metric_delta(
                        args, method, seed, "quadratic_weighted_kappa"
                    ),
                    "delta_mae_vs_raw": float(enhanced_error.mean() - raw_error.mean()),
                    "delta_severe_vs_raw": float(
                        (enhanced_error >= 2).mean() - (raw_error >= 2).mean()
                    ),
                    "corrected_vs_raw": corrected,
                    "broken_vs_raw": broken,
                    "net_correct": corrected - broken,
                    "mcnemar_exact_p": p_value,
                }
            )
            merged["delta_exact_vs_raw"] = enhanced_correct.astype(
                float
            ) - raw_correct.astype(float)
            merged["delta_mae_vs_raw"] = enhanced_error - raw_error
            merged["delta_severe_vs_raw"] = (enhanced_error >= 2).astype(float) - (
                raw_error >= 2
            ).astype(float)
            paired_frames[(method, seed)] = merged
    return pd.DataFrame(rows), paired_frames


def _metric_delta(
    args: argparse.Namespace,
    method: str,
    seed: int,
    metric: str,
) -> float:
    values = {}
    for current_method in ("raw", method):
        path = (
            args.run_root
            / current_method
            / f"seed{seed}"
            / f"{args.report_split}-report"
            / "metrics.json"
        )
        with path.open(encoding="utf-8") as handle:
            values[current_method] = _nested(json.load(handle), METRIC_PATHS[metric])
    return values[method] - values["raw"]


def cluster_bootstrap(
    args: argparse.Namespace,
    paired_frames: dict[tuple[str, int], pd.DataFrame],
) -> pd.DataFrame:
    manifest = pd.read_csv(args.manifest)
    group_column = "split_group_id" if "split_group_id" in manifest.columns else "path"
    lookup = manifest[["path", group_column]].drop_duplicates("path")
    rng = np.random.default_rng(args.bootstrap_seed)
    rows = []
    for method in args.methods:
        if method == "raw":
            continue
        combined = []
        for seed in args.seeds:
            frame = paired_frames[(method, seed)].merge(
                lookup, on="path", how="left", validate="many_to_one"
            )
            frame[group_column] = frame[group_column].fillna(frame["path"])
            combined.append(frame)
        data = pd.concat(combined, ignore_index=True)
        for metric in (
            "delta_exact_vs_raw",
            "delta_mae_vs_raw",
            "delta_severe_vs_raw",
        ):
            grouped = data.groupby(group_column)[metric].agg(["sum", "count"])
            sums = grouped["sum"].to_numpy(dtype=float)
            counts = grouped["count"].to_numpy(dtype=float)
            samples = np.empty(args.bootstrap_samples, dtype=float)
            for index in range(args.bootstrap_samples):
                chosen = rng.integers(0, len(grouped), size=len(grouped))
                samples[index] = sums[chosen].sum() / counts[chosen].sum()
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "point_difference": float(data[metric].mean()),
                    "ci_2_5": float(np.quantile(samples, 0.025)),
                    "ci_97_5": float(np.quantile(samples, 0.975)),
                    "bootstrap_samples": args.bootstrap_samples,
                }
            )
    return pd.DataFrame(rows)


def plot_method_metrics(summary: pd.DataFrame, output: Path) -> None:
    metrics = [
        ("exact_accuracy", "Exact accuracy"),
        ("macro_f1", "Macro F1"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("quadratic_weighted_kappa", "Quadratic weighted kappa"),
        ("mean_absolute_class_error", "Ordinal MAE (lower is better)"),
        (
            "two_or_more_class_error_rate",
            "Severe-error rate (lower is better)",
        ),
        ("ece", "Expected calibration error (lower is better)"),
    ]
    figure, axes = plt.subplots(2, 4, figsize=(20, 9))
    flat_axes = axes.ravel()
    colors = ["#6B7280", "#2563EB", "#D97706", "#059669"][: len(summary)]
    for axis, (metric, title) in zip(flat_axes, metrics, strict=False):
        bars = axis.bar(
            summary["method"],
            summary[f"{metric}_mean"],
            yerr=summary[f"{metric}_std"].fillna(0),
            capsize=4,
            color=colors,
        )
        axis.set_title(title)
        if metric in {
            "exact_accuracy",
            "macro_f1",
            "balanced_accuracy",
            "quadratic_weighted_kappa",
        }:
            axis.set_ylim(0, 1)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    for axis in flat_axes[len(metrics) :]:
        axis.set_visible(False)
    figure.suptitle("Validation performance across seeds (mean ± sample SD)")
    _save(figure, output)


def plot_seed_trajectories(per_seed: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for method, frame in per_seed.groupby("method", sort=False):
        axis.plot(
            frame["seed"].astype(str),
            frame["exact_accuracy"],
            marker="o",
            linewidth=2,
            label=method,
        )
    axis.set_title("Exact validation accuracy by paired seed")
    axis.set_xlabel("Seed")
    axis.set_ylabel("Exact accuracy")
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    _save(figure, output)


def plot_class_recall(per_class: pd.DataFrame, output: Path) -> None:
    recall = (
        per_class.groupby(["method", "class_name"], sort=False)["recall"]
        .mean()
        .unstack("class_name")
        .reindex(columns=CLASS_NAMES)
    )
    figure, axis = plt.subplots(figsize=(11, 5.5))
    recall.plot(kind="bar", ax=axis)
    axis.set_title("Mean validation recall by class")
    axis.set_xlabel("Method")
    axis.set_ylabel("Recall")
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", rotation=0)
    axis.legend(
        [name.replace("_", " ").title() for name in CLASS_NAMES],
        frameon=False,
        ncol=3,
    )
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    _save(figure, output)


def summarize(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = args.output_dir / "figures"
    figures.mkdir(exist_ok=True)
    per_seed, per_class = load_reports(args)
    method_summary = summarize_methods(per_seed)
    class_summary = summarize_classes(per_class)
    paired, paired_frames = paired_comparisons(args)
    bootstrap = cluster_bootstrap(args, paired_frames)

    per_seed.to_csv(args.output_dir / "per_seed_metrics.csv", index=False)
    per_class.to_csv(args.output_dir / "per_seed_class_metrics.csv", index=False)
    method_summary.to_csv(args.output_dir / "method_summary.csv", index=False)
    class_summary.to_csv(args.output_dir / "method_class_summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_vs_raw_per_seed.csv", index=False)
    bootstrap.to_csv(args.output_dir / "cluster_bootstrap_vs_raw.csv", index=False)
    plot_method_metrics(method_summary, figures / "method_metric_comparison.png")
    plot_seed_trajectories(per_seed, figures / "paired_seed_trajectories.png")
    plot_class_recall(per_class, figures / "per_class_recall_comparison.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/final"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/data/manifest.csv")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/generated/validation"),
    )
    parser.add_argument(
        "--report-split",
        choices=("validation", "test"),
        default="validation",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["raw", "clahe", "retinex", "adaptive_gamma"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260729)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "raw" not in args.methods:
        raise ValueError("--methods must include raw for paired comparisons")
    summarize(args)
    print(f"Wrote study summary to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
