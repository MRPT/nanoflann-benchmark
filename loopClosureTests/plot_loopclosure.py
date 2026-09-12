#!/usr/bin/env python3
"""Plot the C5 loop-closure retrieval benchmark (KITTI GT poses, SE(3)).

Reads results/loopclosure.csv + results/perquery_*.csv and writes .pgf into
the paper figs dir and .png previews here. The library under test is
anonymized as "proposed".

Figures:
  * loopclosure_recall.pgf  : recall@8 vs rotation weight, mean over sequences.
  * loopclosure_missmap.pgf : seq-00 top-down trajectory; queries where the
                              naive (sign-continuous) Euclidean tree returns a
                              wrong candidate set are marked.
Also prints the per-sequence LaTeX table body (wrot=15, k=8).
"""
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("pgf")

# Shared paper figure style, kept identical in every plot script so that all
# panels print at the same text size. Figures are generated at their final
# printed width (COL_W for a one-column figure, FULL_W for a figure*) and the
# LaTeX source includes them without \\scalebox: scaling a .pgf scales its text
# too, which is what makes font sizes differ from figure to figure.
COL_W = 3.40   # \\columnwidth in inches
FULL_W = 7.00  # \\textwidth in inches

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.0,
    "pgf.rcfonts": False,
    "figure.autolayout": True,
})

# Okabe-Ito colorblind-safe palette; "proposed" is blue in every figure.
CB = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "gray": "#666666",
}
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_FIGS = os.environ.get("PAPER_FIGS") or os.path.normpath(
    os.path.join(HERE, "..", "..", "..", "papers", "paper-nanoflann",
                 "ieeeral", "figs"))
PREVIEW = os.path.join(HERE, "figs")
os.makedirs(PREVIEW, exist_ok=True)

# (label, color, marker, linestyle); colors from the shared palette and one
# line style per method, so the panels survive grayscale printing.
STYLE = {
    "proposed": ("proposed (exact)", CB["blue"], "o", "-"),
    "naive":    ("naive Eucl.\\ 7-D (canon.)", CB["vermillion"], "^", "--"),
    "naive_sc": ("naive Eucl.\\ 7-D (sign-cont.)", CB["purple"], "v", ":"),
    "rerank":   ("3-D tree + re-rank", CB["green"], "s", "-."),
}


def load():
    rows = []
    with open(os.path.join(HERE, "results", "loopclosure.csv")) as f:
        for r in csv.DictReader(f):
            r["N"] = int(r["N"])
            for c in ("wrot", "build_ms", "query_us", "recall", "top1"):
                r[c] = float(r[c])
            r["k"] = int(r["k"])
            rows.append(r)
    return rows


def savefig(fig, name, w, h):
    fig.set_size_inches(w, h)
    fig.savefig(os.path.join(PAPER_FIGS, name + ".pgf"))
    fig.savefig(os.path.join(PREVIEW, name + ".png"), dpi=160)
    print("wrote", name)


def fig_recall(rows):
    """recall@8 vs rotation weight, mean over sequences."""
    fig, ax = plt.subplots()
    wrots = sorted({r["wrot"] for r in rows})
    for m in ("proposed", "naive", "naive_sc", "rerank"):
        ys = []
        for w in wrots:
            sel = [r["recall"] for r in rows
                   if r["method"] == m and r["k"] == 8 and r["wrot"] == w]
            ys.append(sum(sel) / len(sel))
        lbl, color, marker, ls = STYLE[m]
        ax.plot(wrots, ys, marker=marker, ls=ls, color=color, label=lbl,
                ms=4)
    ax.set_xscale("log")
    ax.set_xticks(wrots)
    ax.set_xticklabels([f"{w:g}" for w in wrots])
    ax.set_xlabel(r"rotation weight $w_R$ [m per unit chordal dist.]")
    ax.set_ylabel("recall@8 vs exact SE(3) NN")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    savefig(fig, "loopclosure_recall", 3.45, 2.3)


def fig_missmap(seq, figname):
    """Trajectory of `seq`; mark queries the naive sign-continuous tree gets
    wrong (wrot=15, k=8)."""
    path = os.path.join(HERE, "results", f"perquery_{seq}_w15_k8.csv")
    if not os.path.exists(path):
        return
    xs, zs, miss = [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            xs.append(float(r["x"]))
            zs.append(float(r["z"]))
            miss.append(float(r["recall_naive_sc"]) < 1.0)
    fig, ax = plt.subplots()
    ax.plot(xs, zs, color=CB["gray"], lw=0.8, zorder=1,
            label=f"trajectory ({seq})")
    mx = [x for x, m in zip(xs, miss) if m]
    mz = [z for z, m in zip(zs, miss) if m]
    ax.scatter(mx, mz, s=4, color=CB["vermillion"], zorder=2,
               label="naive tree:\nwrong candidates")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]" if seq.isdigit() else "y [m]")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    # the map is square, the column is wide: the legend goes in the margin the
    # equal aspect ratio leaves free, instead of on top of the trajectory.
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
              handletextpad=0.5, borderaxespad=0.0)
    n_miss = sum(miss)
    print(f"{seq} misses: {n_miss}/{len(miss)} queries "
          f"({100.0*n_miss/len(miss):.1f}%)")
    savefig(fig, figname, COL_W, 1.75)


def all_missmaps(rows):
    seqs = sorted({r["seq"] for r in rows})
    for seq in seqs:
        # seq "00" keeps the original paper figure name
        name = ("loopclosure_missmap" if seq == "00"
                else f"loopclosure_missmap_{seq}")
        fig_missmap(seq, name)


def table(rows):
    """LaTeX table body: per-sequence summary at wrot=15, k=8."""
    by = defaultdict(dict)
    for r in rows:
        if r["wrot"] == 15 and r["k"] == 8:
            by[r["seq"]][r["method"]] = r
    print("\n% --- LaTeX table body (wrot=15, k=8) ---")
    print("% seq & N & t_BF & t_KD & speedup & build & "
          "recall naive/sign-cont/rerank")
    for seq in sorted(by):
        d = by[seq]
        bf = d["brute"]["query_us"]
        kd = d["proposed"]["query_us"]
        print(f"{seq} & {d['brute']['N']} & {bf:.1f} & {kd:.2f} & "
              f"${bf/kd:.0f}\\times$ & {d['proposed']['build_ms']:.2f} & "
              f"{d['naive']['recall']:.3f} / {d['naive_sc']['recall']:.3f} / "
              f"{d['rerank']['recall']:.3f} \\\\")


def main():
    rows = load()
    fig_recall(rows)
    all_missmaps(rows)
    table(rows)


if __name__ == "__main__":
    main()
