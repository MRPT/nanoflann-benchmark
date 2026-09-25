#!/usr/bin/env python3
"""Figures for the PR #319 review: change vs master (paired median over
frames x repetitions, with the interquartile range as a band) of the original
PR head and of the updated PR, for single KITTI scans and 10-scan local maps."""
import csv
import os
import statistics as st
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = ["rep", "kind", "type", "T", "irep", "frame", "N", "build_ms", "rebuild_ms",
        "knn1_us", "knn10_us", "rad_us", "used_mem", "heap", "nrad", "sink"]
THREADS = [1, 2, 4, 8, 16]

# Reference palette, light surface: categorical slots 1-2, neutral ink.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SERIES = [("final", "updated PR", "#2a78d6"), ("prorig", "PR as submitted", "#eb6834")]


def load(folder, impl):
    with open(os.path.join(HERE, folder, f"static_{impl}.csv")) as f:
        return [dict(zip(COLS, r)) for r in csv.reader(f) if len(r) == len(COLS)]


def paired(folder, impl, metric):
    """{(type, T): [ratio impl/baseline per (rep, frame)]}"""
    base = {(r["rep"], r["type"], r["T"], r["frame"]): float(r[metric])
            for r in load(folder, "baseline")}
    out = defaultdict(list)
    for r in load(folder, impl):
        b = base.get((r["rep"], r["type"], r["T"], r["frame"]))
        if b:
            out[(r["type"], int(r["T"]))].append(float(r[metric]) / b)
    return out


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))]


def figure(metrics, fname, title):
    rows = [("final1x", "single scans (~121k pts)"), ("final10x", "10-scan local maps (~1.2M pts)")]
    fig, axes = plt.subplots(len(rows), len(metrics), figsize=(3.1 * len(metrics), 5.6),
                             sharex=True, facecolor=SURFACE)
    for ri, (folder, rlabel) in enumerate(rows):
        for ci, (metric, typ, mlabel) in enumerate(metrics):
            ax = axes[ri][ci]
            ax.set_facecolor(SURFACE)
            ax.axhline(0, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=1)
            for impl, label, color in SERIES:
                d = paired(folder, impl, metric)
                med = [(st.median(d[(typ, T)]) - 1) * 100 for T in THREADS]
                lo = [(q(d[(typ, T)], 0.25) - 1) * 100 for T in THREADS]
                hi = [(q(d[(typ, T)], 0.75) - 1) * 100 for T in THREADS]
                x = range(len(THREADS))
                ax.fill_between(x, lo, hi, color=color, alpha=0.13, lw=0, zorder=2)
                ax.plot(x, med, color=color, lw=2, marker="o", ms=5.5, zorder=3,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
            ax.set_xticks(range(len(THREADS)))
            ax.set_xticklabels([str(t) for t in THREADS], color=INK2, fontsize=8.5)
            ax.tick_params(axis="y", labelcolor=INK2, labelsize=8.5, length=0)
            ax.tick_params(axis="x", length=0)
            ax.grid(axis="y", color=GRID, lw=0.8)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            # Ticks at 1/2/5 x 10^k only, so whole-percent labels are exact.
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _: f"{v:+.0f}%" if v else "0"))
            if ri == 0:
                ax.set_title(mlabel, color=INK, fontsize=10, pad=8)
            if ci == 0:
                ax.set_ylabel(f"{rlabel}\nvs master (lower is better)", color=INK2,
                              fontsize=8.5)
            if ri == len(rows) - 1:
                ax.set_xlabel("n_thread_build", color=INK2, fontsize=8.5)
    handles, labels = axes[0][0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=INK2, lw=1.2, ls=(0, (4, 3))))
    labels.append("master")
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, color=INK, fontsize=11, y=1.06)
    fig.text(0.5, -0.02, "KITTI 00, leaf_max_size=10. Line: paired median over frames x 5 "
             "repetitions; band: interquartile range.", ha="center", color=INK2, fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", fname), dpi=130, bbox_inches="tight",
                facecolor=SURFACE)
    print("wrote", fname)


figure([("build_ms", "f32", "build, float"), ("build_ms", "f64", "build, double"),
        ("knn1_us", "f32", "kNN k=1, float"), ("knn1_us", "f64", "kNN k=1, double")],
       "pr319_build_knn.png", "Build and kNN time vs master")
figure([("knn10_us", "f32", "kNN k=10, float"), ("knn10_us", "f64", "kNN k=10, double"),
        ("rad_us", "f32", "radius, float"), ("rad_us", "f64", "radius, double")],
       "pr319_queries.png", "Other queries vs master")
