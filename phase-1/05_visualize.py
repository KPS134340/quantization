"""
PHASE 1 STEP 5 — Generate the divergence heatmap.

Produces the central Phase 1 figure:
  Rows    = dataset / category combinations
  Columns = INT8 divergence | INT4 divergence
  Cells   = divergence rate (0–1), color-coded

Outputs:
  figures/phase1_divergence_heatmap_<model>.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import os

os.makedirs("figures", exist_ok=True)

MODEL_KEYS = ["llama3", "mistral"]


def build_heatmap_matrix(agreement_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape the agreement table into a matrix suitable for seaborn heatmap.

    Rows    = "Dataset / Category" labels
    Columns = "INT8 Divergence" and "INT4 Divergence"
    """
    # Create a readable row label
    agreement_df = agreement_df.copy()
    agreement_df["label"] = (
        agreement_df["dataset"] + " / " + agreement_df["category"]
    )

    matrix = agreement_df.set_index("label")[
        ["INT8_divergence", "INT4_divergence"]
    ].rename(columns={
        "INT8_divergence": "INT8\nDivergence",
        "INT4_divergence": "INT4\nDivergence",
    })

    # Sort rows by INT4 divergence descending (highest divergence at top)
    matrix = matrix.sort_values("INT4\nDivergence", ascending=False)
    return matrix


def plot_heatmap(matrix: pd.DataFrame, model_key: str):
    """
    Render and save the heatmap figure.
    """
    fig_height = max(6, len(matrix) * 0.45)
    fig, ax = plt.subplots(figsize=(7, fig_height))

    # Custom red colormap: white at 0, deep red at 1
    cmap = sns.light_palette("crimson", as_cmap=True)

    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="#e0e0e0",
        cbar_kws={"label": "Divergence Rate (vs FP16)", "shrink": 0.6},
        annot_kws={"size": 9},
    )

    # Title and labels
    ax.set_title(
        f"Phase 1 Divergence Heatmap — {model_key.upper()}\n"
        f"(proportion of items where compressed ≠ FP16 output)",
        fontsize=11,
        pad=14,
        fontweight="bold",
    )
    ax.set_xlabel("Precision Condition", fontsize=10)
    ax.set_ylabel("Dataset / Category", fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=9)

    # Add a horizontal reference line at 0.1 (10% divergence threshold)
    # to visually separate low from high divergence rows
    for i, label in enumerate(matrix.index):
        if matrix.iloc[i]["INT4\nDivergence"] > 0.1:
            ax.axhline(y=i + 1, color="navy", linewidth=0.3, alpha=0.4)

    plt.tight_layout()
    out_path = f"figures/phase1_divergence_heatmap_{model_key}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 Saved heatmap → {out_path}")


def plot_category_bar_chart(agreement_df: pd.DataFrame, model_key: str):
    """
    Supplementary bar chart: INT4 divergence per ETHICS category only.
    Useful for the paper's category-level discussion.
    """
    ethics = agreement_df[agreement_df["dataset"] == "ETHICS"].copy()
    if ethics.empty:
        return

    ethics = ethics.sort_values("INT4_divergence", ascending=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [
        "crimson" if cat.lower() == "utilitarianism" else "steelblue"
        for cat in ethics["category"]
    ]
    bars = ax.barh(ethics["category"], ethics["INT4_divergence"], color=colors)
    ax.axvline(x=ethics["INT4_divergence"].mean(), color="grey",
               linestyle="--", linewidth=1, label="Mean divergence")

    ax.set_xlabel("INT4 Divergence Rate (vs FP16)", fontsize=10)
    ax.set_title(
        f"ETHICS Category Divergence — {model_key.upper()} @ INT4",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=9)

    # Annotate bars with values
    for bar, val in zip(bars, ethics["INT4_divergence"]):
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", fontsize=8
        )

    # Note: red = utilitarianism (hypothesis-relevant category)
    ax.text(
        0.98, 0.02,
        "Red = Utilitarianism (hypothesis-relevant)",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7, color="crimson"
    )

    plt.tight_layout()
    out_path = f"figures/phase1_ethics_categories_{model_key}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 Saved category bar chart → {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    for model_key in MODEL_KEYS:
        path = f"data/metrics/{model_key}_agreement_rates.csv"
        if not os.path.exists(path):
            print(f"⚠️  {path} not found — run 04_compute_metrics.py first.")
            continue

        agreement = pd.read_csv(path)
        print(f"\nGenerating figures for {model_key.upper()} ...")

        matrix = build_heatmap_matrix(agreement)
        plot_heatmap(matrix, model_key)
        plot_category_bar_chart(agreement, model_key)

    print("\n✅ Figures complete. Phase 1 pipeline done.")
    print("   Outputs ready for the paper:")
    print("   - figures/phase1_divergence_heatmap_*.png")
    print("   - figures/phase1_ethics_categories_*.png")
    print("   - data/metrics/*_summary.csv  ← key finding sentences")
