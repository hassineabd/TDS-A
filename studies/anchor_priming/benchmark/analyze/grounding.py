"""Aggregate stats and charts for a grounding experiment.

Usage:
    python -m benchmark.analyze.grounding --in results/grounding_zero_shot/raw_predictions.json
    python -m benchmark.analyze.grounding --experiment grounding_with_anchors

Outputs into the same directory as the input JSON:
    summary_by_model.csv
    summary_by_model_size.csv
    charts/error_by_model.png
    charts/hit_rate_by_model.png
    charts/hit_rate_by_size.png
    charts/iou_by_model.png
    charts/latency_by_model.png
    charts/scatter_predictions.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Visual style — consistent palette across all charts; models marketed
# for computer-use distinguished from general-purpose VLMs via hue. (No
# model in this study is actually CU-trained per its provider's primary
# sources — see benchmark/models/openrouter.py docstring.)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
})

MODEL_COLOR = {
    "claude-opus-4.7":   "#0E6D2E",  # darkest green — newest, no-quirk
    "claude-opus-4.6":   "#1F8F3B",
    "claude-sonnet-4.5": "#5DB86C",
    "gpt-4.1":           "#1F5FBF",
    "gpt-4o":            "#6095D6",
    "gemini-2.5-pro":    "#E74C3C",
}
MODEL_ORDER = [
    "claude-opus-4.7", "claude-opus-4.6", "claude-sonnet-4.5",
    "gpt-4.1", "gpt-4o", "gemini-2.5-pro",
]


def _order_models(present: list[str]) -> list[str]:
    return [m for m in MODEL_ORDER if m in present] + [
        m for m in present if m not in MODEL_ORDER
    ]


# Loading

def load_df(path: Path) -> pd.DataFrame:
    rows = json.loads(path.read_text())
    df = pd.json_normalize(rows, sep=".")
    # Surface common metric columns at the top level for ergonomics
    for col in ("metrics.center_error_px", "metrics.hit", "metrics.iou",
                "metrics.parse_fail", "metadata.size_category"):
        short = col.split(".", 1)[1]
        if col in df.columns:
            df[short] = df[col]
    return df


# Aggregations

def summary_by_model(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[~df["parse_fail"].fillna(True)].copy()
    return (valid.groupby("model")
            .agg(n=("center_error_px", "count"),
                 mean_error_px=("center_error_px", "mean"),
                 median_error_px=("center_error_px", "median"),
                 std_error_px=("center_error_px", "std"),
                 hit_rate=("hit", "mean"),
                 mean_iou=("iou", "mean"),
                 mean_latency_ms=("latency_ms", "mean"))
            .round(2).reset_index())


def summary_by_model_size(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[~df["parse_fail"].fillna(True)].copy()
    return (valid.groupby(["model", "size_category"])
            .agg(n=("center_error_px", "count"),
                 mean_error_px=("center_error_px", "mean"),
                 hit_rate=("hit", "mean"),
                 mean_iou=("iou", "mean"))
            .round(2).reset_index())


# Charts

def chart_error_by_model(by_model: pd.DataFrame, out: Path) -> None:
    models = _order_models(by_model["model"].tolist())
    data = by_model.set_index("model").loc[models]
    colors = [MODEL_COLOR.get(m, "#888888") for m in models]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(models, data["mean_error_px"], color=colors,
                  edgecolor="white", linewidth=2)
    ax.errorbar(models, data["mean_error_px"], yerr=data["std_error_px"],
                fmt="none", ecolor="#333", capsize=6, lw=1.5, alpha=0.7)
    for bar, val in zip(bars, data["mean_error_px"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(data["std_error_px"].fillna(0)) * 0.15,
                f"{val:.0f} px", ha="center", fontsize=12, fontweight="bold")

    ax.set_ylabel("Mean center error (pixels)", fontsize=12)
    ax.set_title("Grounding error by model", fontsize=16, pad=12)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=11)
    ax.grid(axis="y", alpha=0.12)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close()


def chart_hit_rate_by_model(by_model: pd.DataFrame, out: Path) -> None:
    models = _order_models(by_model["model"].tolist())
    data = by_model.set_index("model").loc[models]
    colors = [MODEL_COLOR.get(m, "#888888") for m in models]

    fig, ax = plt.subplots(figsize=(11, 6))
    heights = (data["hit_rate"] * 100)
    bars = ax.bar(models, heights, color=colors,
                  edgecolor="white", linewidth=2)
    for bar, val in zip(bars, heights):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.0f}%", ha="center", fontsize=12, fontweight="bold")

    ax.set_ylim(0, 105)
    ax.set_ylabel("Hit rate (% predictions whose center is inside the target)",
                  fontsize=12)
    ax.set_title("Grounding hit rate by model", fontsize=16, pad=12)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=11)
    ax.grid(axis="y", alpha=0.12)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close()


def chart_hit_rate_by_size(by_model_size: pd.DataFrame, out: Path) -> None:
    models = _order_models(by_model_size["model"].unique().tolist())
    sizes_present = sorted(by_model_size["size_category"].unique().tolist(),
                           key=lambda s: ["small", "medium", "large"].index(s)
                           if s in ("small", "medium", "large") else 99)

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_w = 0.13
    x = np.arange(len(sizes_present))
    for i, m in enumerate(models):
        sub = by_model_size[by_model_size["model"] == m].set_index("size_category")
        heights = [float(sub.loc[s, "hit_rate"]) * 100 if s in sub.index else 0
                   for s in sizes_present]
        offset = (i - (len(models) - 1) / 2) * bar_w
        ax.bar(x + offset, heights, width=bar_w,
               color=MODEL_COLOR.get(m, "#888"), label=m,
               edgecolor="white", linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in sizes_present], fontsize=12)
    ax.set_ylabel("Hit rate (%)", fontsize=12)
    ax.set_title("Hit rate by element size", fontsize=16, pad=12)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.12)
    ax.legend(loc="upper left", fontsize=9, frameon=True, fancybox=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close()


def chart_iou_by_model(by_model: pd.DataFrame, out: Path) -> None:
    models = _order_models(by_model["model"].tolist())
    data = by_model.set_index("model").loc[models]
    colors = [MODEL_COLOR.get(m, "#888888") for m in models]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(models, data["mean_iou"], color=colors,
                  edgecolor="white", linewidth=2)
    for bar, val in zip(bars, data["mean_iou"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean IoU", fontsize=12)
    ax.set_title("Bounding-box IoU (predicted vs. ground truth)",
                 fontsize=16, pad=12)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=11)
    ax.grid(axis="y", alpha=0.12)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close()


def chart_latency(by_model: pd.DataFrame, out: Path) -> None:
    models = _order_models(by_model["model"].tolist())
    data = by_model.set_index("model").loc[models]
    colors = [MODEL_COLOR.get(m, "#888888") for m in models]
    fig, ax = plt.subplots(figsize=(11, 5))
    secs = data["mean_latency_ms"] / 1000
    ax.bar(models, secs, color=colors, edgecolor="white", linewidth=2)
    for i, v in enumerate(secs):
        ax.text(i, v + max(secs) * 0.02, f"{v:.1f}s",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean latency (s)", fontsize=12)
    ax.set_title("API latency by model", fontsize=15, pad=10)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=11)
    ax.grid(axis="y", alpha=0.12)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close()


def chart_scatter(df: pd.DataFrame, out: Path) -> None:
    """Scatter of predicted vs ground-truth centers, separately for X and Y.
    Shows where each model systematically over/undershoots."""
    valid = df[~df["parse_fail"].fillna(True)].copy()

    # Extract centers from bboxes (need to reconstruct from list)
    def center_from_bbox(b):
        if b is None or (isinstance(b, float) and pd.isna(b)):
            return None, None
        return ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)

    pred_centers = valid["predicted_bbox_px"].apply(center_from_bbox)
    gt_centers = valid["metadata.ground_truth_bbox_px"].apply(center_from_bbox)
    valid["pred_x"] = pred_centers.apply(lambda c: c[0])
    valid["pred_y"] = pred_centers.apply(lambda c: c[1])
    valid["gt_x"] = gt_centers.apply(lambda c: c[0])
    valid["gt_y"] = gt_centers.apply(lambda c: c[1])
    valid = valid.dropna(subset=["pred_x", "pred_y", "gt_x", "gt_y"])

    models = _order_models(valid["model"].unique().tolist())
    fig, (ax_x, ax_y) = plt.subplots(1, 2, figsize=(14, 6))
    for m in models:
        sub = valid[valid["model"] == m]
        color = MODEL_COLOR.get(m, "#888")
        ax_x.scatter(sub["gt_x"], sub["pred_x"], s=35, alpha=0.6,
                     color=color, label=m, edgecolors="white", linewidth=0.7)
        ax_y.scatter(sub["gt_y"], sub["pred_y"], s=35, alpha=0.6,
                     color=color, label=m, edgecolors="white", linewidth=0.7)

    for ax, axis in ((ax_x, "X"), (ax_y, "Y")):
        col = f"gt_{axis.lower()}"
        mx = max(valid[col].max(),
                 valid[f"pred_{axis.lower()}"].max())
        ax.plot([0, mx], [0, mx], "--", color="#888", alpha=0.5, lw=1)
        ax.set_xlabel(f"Ground truth {axis} center (px)", fontsize=11)
        ax.set_ylabel(f"Predicted {axis} center (px)", fontsize=11)
        ax.grid(alpha=0.1)
    ax_x.set_title("X coordinates", fontsize=14)
    ax_y.set_title("Y coordinates", fontsize=14)
    fig.suptitle("Predicted vs ground-truth centers (diagonal = perfect)",
                 fontsize=15, fontweight="bold", y=1.02)
    ax_x.legend(loc="upper left", fontsize=8, frameon=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close()


# Entry point

def latest_run_dir(experiment: str, results_root: Path = Path("results")) -> Path | None:
    """Return the most recent <results_root>/<experiment>/<timestamp_*> dir.

    Sorted by directory name (lexicographic — works because we use
    YYYYMMDD-HHMMSS prefix), not by mtime, so renames don't reorder.
    Returns None if no subdirectory contains a raw_predictions.json.
    """
    exp_dir = results_root / experiment
    if not exp_dir.is_dir():
        return None
    candidates = [
        d for d in exp_dir.iterdir()
        if d.is_dir() and (d / "raw_predictions.json").exists()
    ]
    if not candidates:
        return None
    return sorted(candidates)[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", type=Path, default=None,
                        help="Path to raw_predictions.json")
    parser.add_argument("--experiment", default=None,
                        help="Shortcut: pick the most recent run dir under "
                             "results/<experiment>/")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Explicit run directory, e.g. "
                             "results/grounding_zero_shot/20260512-022021_baseline-6models")
    args = parser.parse_args()

    if args.run_dir is not None:
        args.inp = args.run_dir / "raw_predictions.json"
    elif args.inp is None:
        if args.experiment is None:
            parser.error("Provide --in, --run-dir, or --experiment")
        latest = latest_run_dir(args.experiment)
        if latest is None:
            print(f"ERROR: no runs found under results/{args.experiment}/",
                  file=sys.stderr)
            return
        args.inp = latest / "raw_predictions.json"
        print(f"Using most recent run: {latest}")

    if not args.inp.exists():
        print(f"ERROR: {args.inp} does not exist")
        return

    out_dir = args.inp.parent
    df = load_df(args.inp)
    n_total = len(df)
    n_fail = df["parse_fail"].fillna(True).sum()
    print(f"Loaded {n_total} rows from {args.inp}")
    print(f"  Parse failures: {n_fail}  ({n_fail/n_total*100:.1f}%)")

    by_model = summary_by_model(df)
    by_model_size = summary_by_model_size(df)

    print("\n=== Per model ===")
    print(by_model.to_string(index=False))
    print("\n=== Per model x size ===")
    print(by_model_size.to_string(index=False))

    by_model.to_csv(out_dir / "summary_by_model.csv", index=False)
    by_model_size.to_csv(out_dir / "summary_by_model_size.csv", index=False)

    charts_dir = out_dir / "charts"
    chart_error_by_model(by_model, charts_dir / "error_by_model.png")
    chart_hit_rate_by_model(by_model, charts_dir / "hit_rate_by_model.png")
    chart_hit_rate_by_size(by_model_size, charts_dir / "hit_rate_by_size.png")
    chart_iou_by_model(by_model, charts_dir / "iou_by_model.png")
    chart_latency(by_model, charts_dir / "latency_by_model.png")
    chart_scatter(df, charts_dir / "scatter_predictions.png")
    print(f"\nCharts in {charts_dir}/")


if __name__ == "__main__":
    main()
