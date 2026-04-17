"""Statistical analysis + charts from raw_predictions.json.

Outputs:
    results/summary.csv                 - per model x size_category stats
    results/summary_by_model.csv        - per model overall
    results/charts/error_by_model.png   - bar chart of mean error
    results/charts/hit_rate_by_size.png - grouped bar by size
    results/charts/scatter_predictions.png - scatter per model
    results/charts/latency_by_model.png - bar chart latency
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Visual style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
})

# Color per model - green/blue for CU-trained, red for not
MODEL_COLOR = {
    "claude-opus-4.5":   "#1F8F3B",
    "claude-sonnet-4.5": "#5DB86C",
    "gpt-4.1":           "#1F5FBF",
    "gpt-4o":            "#6095D6",
    "gemini-2.5-pro":    "#E74C3C",
}
MODEL_ORDER = ["claude-opus-4.5", "claude-sonnet-4.5", "gpt-4.1", "gpt-4o", "gemini-2.5-pro"]


def load_df(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text())
    df = pd.DataFrame(data)
    # Drop parse failures (error != None AND predicted is -1)
    df["parse_fail"] = df["predicted"].apply(lambda p: p[0] < 0)
    return df


def summary_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = df[~df["parse_fail"]].copy()

    by_model = (valid.groupby("model")
                .agg(n=("euclidean_error_px", "count"),
                     mean_error=("euclidean_error_px", "mean"),
                     median_error=("euclidean_error_px", "median"),
                     std_error=("euclidean_error_px", "std"),
                     hit_rate=("inside_bounds", "mean"),
                     mean_latency_ms=("latency_ms", "mean"))
                .round(1).reset_index())

    by_model_size = (valid.groupby(["model", "size_category"])
                     .agg(n=("euclidean_error_px", "count"),
                          mean_error=("euclidean_error_px", "mean"),
                          hit_rate=("inside_bounds", "mean"))
                     .round(1).reset_index())
    return by_model, by_model_size


def _order_models(present: list[str]) -> list[str]:
    return [m for m in MODEL_ORDER if m in present] + [
        m for m in present if m not in MODEL_ORDER
    ]


def chart_error_by_model(by_model: pd.DataFrame, out: Path) -> None:
    models = _order_models(by_model["model"].tolist())
    data = by_model.set_index("model").loc[models]
    colors = [MODEL_COLOR.get(m, "#888888") for m in models]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(models, data["mean_error"], color=colors,
                  edgecolor="white", linewidth=2)
    # Error bars = std
    ax.errorbar(models, data["mean_error"], yerr=data["std_error"],
                fmt="none", ecolor="#333", capsize=6, lw=1.5, alpha=0.7)

    for bar, val, n in zip(bars, data["mean_error"], data["n"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(data["std_error"]) * 0.15,
                f"{val:.0f} px", ha="center", fontsize=13, fontweight="bold")

    ax.set_ylabel("Mean Euclidean Error (pixels)", fontsize=13)
    ax.set_title("Mobile UI grounding error by model", fontsize=18, pad=15)
    ax.set_xticklabels(models, rotation=12, ha="right", fontsize=12)
    ax.grid(axis="y", alpha=0.12)

    # Legend showing CU-trained vs not
    cu_patch = mpatches.Patch(color="#5DB86C", label="CU-trained (Claude, GPT)")
    nocu_patch = mpatches.Patch(color="#E74C3C", label="Not CU-trained (Gemini)")
    ax.legend(handles=[cu_patch, nocu_patch], loc="upper left",
              frameon=True, fancybox=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close()


def chart_hit_rate_by_size(by_model_size: pd.DataFrame, out: Path) -> None:
    models = _order_models(by_model_size["model"].unique().tolist())
    sizes = ["small", "medium", "large"]

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_w = 0.15
    x = np.arange(len(sizes))

    for i, m in enumerate(models):
        sub = by_model_size[by_model_size["model"] == m].set_index("size_category")
        heights = [float(sub.loc[s, "hit_rate"]) * 100 if s in sub.index else 0
                   for s in sizes]
        offset = (i - (len(models) - 1) / 2) * bar_w
        ax.bar(x + offset, heights, width=bar_w,
               color=MODEL_COLOR.get(m, "#888"), label=m,
               edgecolor="white", linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in sizes], fontsize=12)
    ax.set_ylabel("Hit rate (% predictions inside element bounds)", fontsize=12)
    ax.set_title("Hit rate by element size", fontsize=18, pad=15)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.12)
    ax.legend(loc="upper left", frameon=True, fancybox=True, fontsize=10)

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close()


def chart_latency(by_model: pd.DataFrame, out: Path) -> None:
    models = _order_models(by_model["model"].tolist())
    data = by_model.set_index("model").loc[models]
    colors = [MODEL_COLOR.get(m, "#888888") for m in models]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(models, data["mean_latency_ms"] / 1000, color=colors,
           edgecolor="white", linewidth=2)
    for i, v in enumerate(data["mean_latency_ms"] / 1000):
        ax.text(i, v + 0.1, f"{v:.1f}s", ha="center", fontsize=12,
                fontweight="bold")
    ax.set_ylabel("Mean latency (seconds)", fontsize=13)
    ax.set_title("API latency by model", fontsize=16, pad=12)
    ax.set_xticklabels(models, rotation=12, ha="right", fontsize=12)
    ax.grid(axis="y", alpha=0.12)

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close()


def chart_scatter_predictions(df: pd.DataFrame, out: Path) -> None:
    """Scatter: ground truth X vs predicted X, one plot per axis with per-model color."""
    valid = df[~df["parse_fail"]].copy()
    valid["gt_x"] = valid["ground_truth"].apply(lambda c: c[0])
    valid["gt_y"] = valid["ground_truth"].apply(lambda c: c[1])
    valid["pred_x"] = valid["predicted"].apply(lambda c: c[0])
    valid["pred_y"] = valid["predicted"].apply(lambda c: c[1])

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
        mx = max(valid[f"gt_{axis.lower()}"].max(),
                 valid[f"pred_{axis.lower()}"].max())
        ax.plot([0, mx], [0, mx], "--", color="#888", alpha=0.5, lw=1)
        ax.set_xlabel(f"Ground truth {axis} (px)", fontsize=12)
        ax.set_ylabel(f"Predicted {axis} (px)", fontsize=12)
        ax.grid(alpha=0.1)

    ax_x.set_title(f"X coordinates", fontsize=14)
    ax_y.set_title(f"Y coordinates", fontsize=14)
    fig.suptitle("Predicted vs ground truth coordinates (diagonal = perfect)",
                 fontsize=16, fontweight="bold", y=1.02)
    ax_x.legend(loc="upper left", fontsize=9, frameon=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", type=Path,
                        default=Path("results/raw_predictions.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    df = load_df(args.inp)
    print(f"Loaded {len(df)} predictions; "
          f"{df['parse_fail'].sum()} parse failures.")

    by_model, by_model_size = summary_stats(df)

    print("\n=== Per model ===")
    print(by_model.to_string(index=False))
    print("\n=== Per model x size ===")
    print(by_model_size.to_string(index=False))

    by_model.to_csv(args.out_dir / "summary_by_model.csv", index=False)
    by_model_size.to_csv(args.out_dir / "summary.csv", index=False)

    charts_dir = args.out_dir / "charts"
    chart_error_by_model(by_model, charts_dir / "error_by_model.png")
    chart_hit_rate_by_size(by_model_size, charts_dir / "hit_rate_by_size.png")
    chart_latency(by_model, charts_dir / "latency_by_model.png")
    chart_scatter_predictions(df, charts_dir / "scatter_predictions.png")
    print(f"\nCharts saved in {charts_dir}/")


if __name__ == "__main__":
    main()
