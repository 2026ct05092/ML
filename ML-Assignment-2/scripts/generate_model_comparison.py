"""Generate consolidated visual comparisons for the five classifiers.

Run from the repository root:
    python scripts/generate_model_comparison.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = ROOT / "results" / "model_comparison.csv"
OUTPUT_DIR = ROOT / "results" / "visualizations"

METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def main() -> None:
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Could not find comparison results: {RESULTS_FILE}")

    results = pd.read_csv(RESULTS_FILE)
    required = {"Model", *METRICS}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Comparison CSV is missing columns: {sorted(missing)}")
    if results.empty or results["Model"].nunique() != 5:
        raise ValueError("Expected results for exactly five distinct classification models.")
    if results[METRICS].isna().any().any():
        raise ValueError("Metric values contain missing entries.")
    if ((results[METRICS] < 0) | (results[METRICS] > 1)).any().any():
        raise ValueError("Expected all six classification metrics to be between 0 and 1.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = results["Model"].tolist()

    # Six-panel figure: one common scale makes metric leaders easy to compare.
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    for ax, metric in zip(axes.flat, METRICS):
        values = results[metric].to_numpy()
        bars = ax.bar(range(len(labels)), values)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Score")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018,
                    f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Classification Model Comparison – Held-out Test Set", fontsize=17, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "all_models_metrics_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Standalone plots can be inserted individually into the assignment PDF.
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(9, 5.2))
        values = results[metric].to_numpy()
        bars = ax.bar(labels, values)
        ax.set_title(f"{metric} Comparison Across Five Models", fontsize=14, fontweight="bold")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.08)
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018,
                    f"{value:.3f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"{metric.lower()}_comparison.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    print(f"Validated {len(results)} models and {len(METRICS)} metrics.")
    print(f"Saved consolidated chart: {OUTPUT_DIR / 'all_models_metrics_comparison.png'}")
    print(f"Saved six individual metric charts under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
