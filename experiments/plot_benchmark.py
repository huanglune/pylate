"""Visualize benchmark results: PLAID vs WARP across BEIR datasets."""

from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import numpy as np

RESULT_FILE = (
    "results/benchmark_plaid+warp_nfcorpus+scifact+arguana+scidocs+fiqa+trec-covid_fp16_20260530_114527.json"
)


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else RESULT_FILE
    data = load_results(path)

    datasets = ["nfcorpus", "scifact", "arguana", "scidocs", "fiqa", "trec-covid"]
    metrics_to_plot = ["ndcg@10", "recall@100", "map"]

    plaid = {ds: data[f"{ds}/plaid"] for ds in datasets}
    warp = {ds: data[f"{ds}/warp"] for ds in datasets}

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("PLAID vs WARP Benchmark (fp16, GTE-ModernColBERT-v1)", fontsize=16, fontweight="bold")

    x = np.arange(len(datasets))
    bar_width = 0.35

    colors_plaid = "#4C72B0"
    colors_warp = "#DD8452"

    # --- Row 1: 3 metric subplots ---
    for i, metric in enumerate(metrics_to_plot):
        ax = fig.add_subplot(2, 3, i + 1)
        vals_p = [plaid[ds]["metrics"][metric] for ds in datasets]
        vals_w = [warp[ds]["metrics"][metric] for ds in datasets]

        bars_p = ax.bar(x - bar_width / 2, vals_p, bar_width, label="PLAID", color=colors_plaid, edgecolor="white")
        bars_w = ax.bar(x + bar_width / 2, vals_w, bar_width, label="WARP", color=colors_warp, edgecolor="white")

        for bar, val in zip(bars_p, vals_p):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)
        for bar, val in zip(bars_w, vals_w):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_title(metric.upper(), fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=25, ha="right", fontsize=9)
        ax.set_ylim(0, max(max(vals_p), max(vals_w)) * 1.18)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # --- Row 2 Left: QPS comparison ---
    ax_qps = fig.add_subplot(2, 3, 4)
    qps_p = [plaid[ds]["timing"]["queries_per_second"] for ds in datasets]
    qps_w = [warp[ds]["timing"]["queries_per_second"] for ds in datasets]

    bars_p = ax_qps.bar(x - bar_width / 2, qps_p, bar_width, label="PLAID", color=colors_plaid, edgecolor="white")
    bars_w = ax_qps.bar(x + bar_width / 2, qps_w, bar_width, label="WARP", color=colors_warp, edgecolor="white")

    for bar, val in zip(bars_p, qps_p):
        ax_qps.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars_w, qps_w):
        ax_qps.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=8)

    ax_qps.set_title("Queries Per Second (QPS)", fontsize=13, fontweight="bold")
    ax_qps.set_xticks(x)
    ax_qps.set_xticklabels(datasets, rotation=25, ha="right", fontsize=9)
    ax_qps.set_ylabel("QPS")
    ax_qps.legend(fontsize=9)
    ax_qps.grid(axis="y", alpha=0.3)
    ax_qps.spines["top"].set_visible(False)
    ax_qps.spines["right"].set_visible(False)

    # --- Row 2 Center: Latency p50 ---
    ax_lat = fig.add_subplot(2, 3, 5)
    lat_p = [plaid[ds]["timing"]["latency_ms"]["p50"] for ds in datasets]
    lat_w = [warp[ds]["timing"]["latency_ms"]["p50"] for ds in datasets]

    bars_p = ax_lat.bar(x - bar_width / 2, lat_p, bar_width, label="PLAID", color=colors_plaid, edgecolor="white")
    bars_w = ax_lat.bar(x + bar_width / 2, lat_w, bar_width, label="WARP", color=colors_warp, edgecolor="white")

    for bar, val in zip(bars_p, lat_p):
        ax_lat.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars_w, lat_w):
        ax_lat.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax_lat.set_title("Latency p50 (ms)", fontsize=13, fontweight="bold")
    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels(datasets, rotation=25, ha="right", fontsize=9)
    ax_lat.set_ylabel("ms")
    ax_lat.legend(fontsize=9)
    ax_lat.grid(axis="y", alpha=0.3)
    ax_lat.spines["top"].set_visible(False)
    ax_lat.spines["right"].set_visible(False)

    # --- Row 2 Right: WARP speedup ratio ---
    ax_speedup = fig.add_subplot(2, 3, 6)
    speedup = [qps_w[i] / qps_p[i] for i in range(len(datasets))]

    bars = ax_speedup.bar(x, speedup, bar_width * 1.5, color="#55A868", edgecolor="white")
    for bar, val in zip(bars, speedup):
        ax_speedup.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f"{val:.1f}×", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax_speedup.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax_speedup.set_title("WARP Speedup over PLAID", fontsize=13, fontweight="bold")
    ax_speedup.set_xticks(x)
    ax_speedup.set_xticklabels(datasets, rotation=25, ha="right", fontsize=9)
    ax_speedup.set_ylabel("Speedup (×)")
    ax_speedup.grid(axis="y", alpha=0.3)
    ax_speedup.spines["top"].set_visible(False)
    ax_speedup.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = "results/benchmark_plaid_vs_warp_fp16.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
