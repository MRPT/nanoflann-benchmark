#!/usr/bin/env python3
"""
Baseline vs candidate nanoflann concurrent-build benchmark.
Adapted from nanoflann-benchmark/plot_baseline_vs_new_kitti.py.
Reads results/kitti00_{master|pr318}_T{N}.txt and writes bar charts into figures/.
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 12,
    "figure.dpi": 150,
})

# categorical slots 1 and 2 of the validated default palette
COLORS = {"master": "#2a78d6", "pr318": "#eb6834"}
LABELS = {"master": "master (29e8e26)", "pr318": "PR #318"}
INK, INK_MUTED = "#1a1a19", "#6b6a63"

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")
os.makedirs(FIGURES, exist_ok=True)

data = {}
for f in sorted(glob.glob(os.path.join(RESULTS, "kitti00_*_T*.txt"))):
    parts = os.path.basename(f)[:-4].split("_")
    method = parts[1]
    thread = int(parts[2][1:])
    df = pd.read_csv(f, sep=r"\s+", names=["NUM_POINTS", "BUILD", "QUERY"])
    d = {}
    for m in ("BUILD", "QUERY"):
        mean = df[m].mean()
        d[f"{m}_MEAN"] = mean
        d[f"{m}_ERR_LOW"] = mean - df[m].quantile(0.10)
        d[f"{m}_ERR_HIGH"] = df[m].quantile(0.90) - mean
    d["N_FRAMES"] = len(df)
    d["PTS"] = df["NUM_POINTS"].mean()
    data[(thread, method)] = d

threads = sorted({t for (t, _) in data})
n_samples = max(v["N_FRAMES"] for v in data.values())
pts = max(v["PTS"] for v in data.values())
WIDTH = 0.36


def make_chart(metric, title_suffix, filename, scale, unit):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(threads))

    series = {}
    for method, off in (("master", -WIDTH / 2), ("pr318", +WIDTH / 2)):
        xs = x + off
        means = np.array([data[(t, method)][f"{metric}_MEAN"] for t in threads]) * scale
        lo = np.array([data[(t, method)][f"{metric}_ERR_LOW"] for t in threads]) * scale
        hi = np.array([data[(t, method)][f"{metric}_ERR_HIGH"] for t in threads]) * scale
        series[method] = (xs, means, hi)
        ax.bar(xs, means, width=WIDTH - 0.02, yerr=[lo, hi], capsize=4,
               color=COLORS[method], label=LABELS[method],
               error_kw={"elinewidth": 1.2, "ecolor": INK_MUTED})

    # direct labels on every bar (2 series, few bars: identity is never color-alone)
    for method in ("master", "pr318"):
        xs, means, hi = series[method]
        for xi, m, h in zip(xs, means, hi):
            ax.text(xi, m + h * 1.05 + 0.012 * means.max(), f"{m:.3g}",
                    ha="center", va="bottom", fontsize=9, color=INK_MUTED)

    # relative change, PR vs master, above each pair
    top = 0.0
    for i, t in enumerate(threads):
        bm = series["master"][1][i]
        nm = series["pr318"][1][i]
        pct = (nm - bm) / bm * 100.0
        y = max(bm + series["master"][2][i], nm + series["pr318"][2][i]) * 1.10
        ax.text(i, y, f"{pct:+.0f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold",
                color=(INK_MUTED if abs(pct) < 1.0
                       else "#0d6b2f" if pct < 0 else "#b3201f"))
        top = max(top, y)

    ax.set_ylim(0, top * 1.16)
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in threads])
    ax.set_xlabel("n_thread_build   (16 = hardware_concurrency on this machine)")
    ax.set_ylabel(f"Mean {unit}   ·   10-90th pct error bars")
    ax.set_title(
        f"nanoflann concurrent KD-tree build · {title_suffix}\n"
        f"KITTI odometry seq-00 · {n_samples} builds/config · ~{pts/1000:.0f}k pts · "
        f"leaf_max_size=10 · AMD Ryzen 7 2700X (8C/16T)",
        pad=12, color=INK)
    ax.legend(loc="upper left", frameon=False)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
    ax.grid(axis="y", linestyle="--", alpha=0.3, color=INK_MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK_MUTED)
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight", facecolor="white")
    print("Saved:", filename)
    plt.close(fig)


if not data:
    sys.exit(f"no result files in {RESULTS}; run run_kitti.sh first")

make_chart("BUILD", "index build time (lower is better)",
           os.path.join(FIGURES, "pr318_build_time.png"), 1e3, "build time (ms)")
make_chart("QUERY", "single-point kNN query time (lower is better)",
           os.path.join(FIGURES, "pr318_query_time.png"), 1e6, "query time (us)")

# ---- markdown table ----
s1 = data[(1, "pr318")]["BUILD_MEAN"]
lines = ["| threads | master build | PR #318 build | PR vs master | PR speedup vs own serial "
         "| master query | PR #318 query |",
         "|---:|---:|---:|---:|---:|---:|---:|"]
for t in threads:
    m, p = data[(t, "master")], data[(t, "pr318")]
    lines.append(
        f"| {t} | {m['BUILD_MEAN']*1e3:.2f} ms | {p['BUILD_MEAN']*1e3:.2f} ms "
        f"| **{m['BUILD_MEAN']/p['BUILD_MEAN']:.2f}x** | {s1/p['BUILD_MEAN']:.2f}x "
        f"| {m['QUERY_MEAN']*1e6:.3f} us | {p['QUERY_MEAN']*1e6:.3f} us |")
open(os.path.join(RESULTS, "table.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
