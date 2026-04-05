"""
dashboard.py — Sentiment results visualisation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import config

COLORS = {"positive": "#2ecc71", "neutral": "#3498db", "negative": "#e74c3c"}


def plot_results(eval_results: dict, predictions: list, df_val=None) -> None:
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle("Financial Sentiment Analysis — DistilBERT Fine-Tuning Results",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    # ── Metric bar chart ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    metrics = {k: v for k, v in eval_results.items() if isinstance(v, float)}
    keys    = [k.replace("eval_", "") for k in metrics]
    vals    = list(metrics.values())
    bars = ax1.bar(keys, vals, color=["#3498db", "#2ecc71", "#e74c3c", "#f39c12"][:len(vals)],
                   alpha=0.85, width=0.5)
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{v:.4f}", ha="center", fontsize=9)
    ax1.set_title("Evaluation Metrics", fontweight="bold")
    ax1.set_ylim(0, 1.15)
    ax1.axhline(0.72, color="gray", linestyle="--", alpha=0.5,
                label="Baseline (zero-shot)")
    ax1.legend(fontsize=8)

    # ── Prediction distribution ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    if predictions:
        sents  = [p["sentiment"].lower() for p in predictions]
        unique = list(COLORS.keys())
        counts = [sents.count(s) for s in unique]
        wedges, texts, autotexts = ax2.pie(
            counts, labels=[s.capitalize() for s in unique], autopct="%1.0f%%",
            colors=[COLORS[s] for s in unique], startangle=90
        )
        for at in autotexts:
            at.set_fontsize(9)
    ax2.set_title("Demo Prediction Distribution", fontweight="bold")

    # ── Confidence distribution ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    if predictions:
        confs = [p["confidence"] for p in predictions]
        ax3.hist(confs, bins=10, color="#9b59b6", alpha=0.8, edgecolor="white")
        ax3.axvline(np.mean(confs), color="#e74c3c", linewidth=2,
                    linestyle="--", label=f"Mean = {np.mean(confs):.3f}")
        ax3.set_title("Prediction Confidence", fontweight="bold")
        ax3.set_xlabel("Confidence Score")
        ax3.set_ylabel("Count")
        ax3.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(config.CHART_OUTPUT, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"Dashboard saved → {config.CHART_OUTPUT}")
