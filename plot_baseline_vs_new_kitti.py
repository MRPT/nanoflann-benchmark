#!/bin/env python3
"""
Plot baseline (v1.9.0) vs new (v1.10.0) nanoflann benchmark results.
Reads stats_kitti_00_{baseline|new}_T{N}.txt files and produces
build-time and query-time bar charts saved as PNG.
"""

import os
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 12,
    "figure.dpi": 150,
})

COLORS = {
    "baseline": "#4C72B0",   # muted blue  → v1.9.0
    "new":      "#DD8452",   # warm orange → v1.10.0
}
LABELS = {
    "baseline": "v1.9.0 (baseline)",
    "new":      "v1.10.0 (new)",
}


def annotate_percent_change(ax, x_baseline, x_new, baseline_mean, baseline_err_high,
                            new_mean, new_err_high, y_offset=0.05):
    max_y = 0.0
    for xb, xn, b_m, b_eh, n_m, n_eh in zip(
        x_baseline, x_new, baseline_mean, baseline_err_high, new_mean, new_err_high
    ):
        if b_m == 0:
            continue
        pct = (n_m - b_m) / b_m * 100.0
        top = max(b_m + b_eh, n_m + n_eh)
        y = top * (1 + y_offset)
        xc = (xb + xn) / 2
        color = "#228B22" if pct <= 0 else "#CC2200"
        ax.text(xc, y, f"{pct:+.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=color)
        max_y = max(max_y, y)
    return max_y


# ── Load data ──────────────────────────────────────────────────────────────────
files = glob.glob("stats_kitti_*.txt")

data = {}
for f in files:
    base = os.path.basename(f).replace(".txt", "")
    parts = base.split("_")
    method = "new" if "new" in parts else "baseline"
    thread_str = [p for p in parts if p.startswith("T")][-1]
    thread = int(thread_str[1:])

    df = pd.read_csv(f, sep=r"\s+", names=["NUM_POINTS", "BUILD", "QUERY"])

    bm = df["BUILD"].mean()
    qm = df["QUERY"].mean()
    data[(thread, method)] = {
        "BUILD_MEAN":     bm,
        "BUILD_ERR_LOW":  bm - df["BUILD"].quantile(0.10),
        "BUILD_ERR_HIGH": df["BUILD"].quantile(0.90) - bm,
        "QUERY_MEAN":     qm,
        "QUERY_ERR_LOW":  qm - df["QUERY"].quantile(0.10),
        "QUERY_ERR_HIGH": df["QUERY"].quantile(0.90) - qm,
        "N_FRAMES":       len(df),
    }

threads = sorted(set(t for (t, _) in data.keys()))
records = []
for thread in threads:
    for method in ["baseline", "new"]:
        if (thread, method) not in data:
            continue
        records.append({"Thread": thread, "Method": method, **data[(thread, method)]})

df_all = pd.DataFrame(records)
n_frames = int(df_all["N_FRAMES"].max())

WIDTH = 0.35


def make_bar_chart(metric, ylabel, title_suffix, filename, unit_scale=1.0, unit_label="s"):
    fig, ax = plt.subplots(figsize=(10, 5.5))

    x_positions = np.arange(len(threads))

    x_b_list, x_n_list = [], []
    b_means, b_ehl, b_ehh = [], [], []
    n_means, n_ehl, n_ehh = [], [], []

    for i, t in enumerate(threads):
        for method, offset, xlist, ml, ehl, ehh in [
            ("baseline", -WIDTH/2, x_b_list, b_means, b_ehl, b_ehh),
            ("new",       +WIDTH/2, x_n_list, n_means, n_ehl, n_ehh),
        ]:
            row = df_all[(df_all["Thread"] == t) & (df_all["Method"] == method)]
            if row.empty:
                continue
            xlist.append(i + offset)
            ml.append(row[f"{metric}_MEAN"].values[0] * unit_scale)
            ehl.append(row[f"{metric}_ERR_LOW"].values[0] * unit_scale)
            ehh.append(row[f"{metric}_ERR_HIGH"].values[0] * unit_scale)

    for method, xlist, ml, ehl, ehh in [
        ("baseline", x_b_list, b_means, b_ehl, b_ehh),
        ("new",       x_n_list, n_means, n_ehl, n_ehh),
    ]:
        ax.bar(xlist, ml,
               width=WIDTH,
               yerr=[ehl, ehh],
               capsize=5,
               color=COLORS[method],
               label=LABELS[method],
               alpha=0.88,
               error_kw={"elinewidth": 1.5, "ecolor": "#333333"})

    max_y = annotate_percent_change(
        ax, x_b_list, x_n_list,
        b_means, b_ehh, n_means, n_ehh,
    )

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, max(ymax, max_y * 1.12))

    hw = 16  # hardware_concurrency on benchmark machine
    thread_labels = [f"auto\n({hw} cores)" if t == 0 else str(t) for t in threads]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(thread_labels)
    ax.set_xlabel("Number of threads  (auto = all cores detected at runtime)")
    ax.set_ylabel(f"Mean time ({unit_label})  ·  10–90th pct error bars")
    ax.set_title(
        f"nanoflann v1.9.0 vs v1.10.0  ·  {title_suffix}\n"
        f"KITTI odometry seq-00  ·  {n_frames} LiDAR frames  ·  ~110k pts/frame",
        pad=10,
    )
    ax.legend(loc="upper right")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close(fig)


make_bar_chart("BUILD", "Build time (s)", "Index build time",
               "benchmark_build_time.png")
make_bar_chart("QUERY", "Query time (s/query)", "Single-point kNN query time",
               "benchmark_query_time.png")

print("Done.")
