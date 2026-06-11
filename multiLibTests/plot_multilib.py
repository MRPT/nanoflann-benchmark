#!/usr/bin/env python3
"""Plot the multi-library single-tree build/query benchmark.

Reads results/multilib_bench.csv (produced by bench.cpp) and writes:
  * .pgf into the paper's figure dir (for \\import in LaTeX),
  * .png previews into this repo (for the README / quick inspection).

The library under test is anonymized as "proposed" (double-blind RA-L).
"""
import csv
import os

import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pgf.rcfonts": False,
    "figure.autolayout": True,
})
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_FIGS = os.path.normpath(
    os.path.join(HERE, "..", "..", "..", "papers", "nanoflann-paper",
                 "ieeeral", "figs"))
PREVIEW = os.path.join(HERE, "figs")
os.makedirs(PREVIEW, exist_ok=True)

# Display order, pretty label, color, marker. fastann-exact is a linear scan,
# so we treat it as a brute-force-class baseline and omit it from the KD-tree
# comparison (it tracks brute_force exactly).
STYLE = {
    "proposed":     ("proposed",       "#1f6fb2", "o", "-"),
    "pico_tree":    ("PicoTree",        "#2c8c3c", "s", "-"),
    "flann_kdtree": ("FLANN (exact)",   "#d1622b", "^", "-"),
    "pcl_kdtree":   ("PCL KdTreeFLANN", "#7a4fb5", "D", "-"),
    "libnabo":      ("libnabo",         "#c4a000", "v", "-"),
    "libkdtree":    ("libkdtree++",     "#8c564b", "P", "-"),
    "flann_4rand":  ("FLANN (4 rand., approx)", "#e377c2", "X", "--"),
    "hnsw_approx":  ("HNSW (approx)",   "#17becf", "*", "--"),
    "brute_force":  ("brute force",     "0.45",    "",  ":"),
}


def load():
    rows = []
    with open(os.path.join(HERE, "results", "multilib_bench.csv")) as f:
        for r in csv.DictReader(f):
            r["dim"] = int(r["dim"])
            r["N"] = int(r["N"])
            for c in ("build_ms", "query_us", "recall"):
                r[c] = float(r[c])
            rows.append(r)
    return rows


def series(rows, dataset, lib, xkey, ykey):
    pts = sorted([r for r in rows if r["dataset"] == dataset
                  and r["library"] == lib], key=lambda r: r[xkey])
    return [r[xkey] for r in pts], [r[ykey] for r in pts]


def save(fig, name):
    fig.savefig(os.path.join(PAPER_FIGS, name + ".pgf"))
    fig.savefig(os.path.join(PREVIEW, name + ".png"), dpi=160)
    plt.close(fig)


def fig_kitti(rows):
    libs = ["proposed", "pico_tree", "flann_kdtree", "pcl_kdtree",
            "libnabo", "libkdtree"]
    # (a) build time vs N
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for lib in libs:
        x, y = series(rows, "kitti", lib, "N", "build_ms")
        if not x:
            continue
        lab, col, mk, ls = STYLE[lib]
        ax.loglog(x, y, ls=ls, marker=mk, ms=3.5, color=col, label=lab)
    ax.set_xlabel("dataset size $N$ (KITTI LiDAR points)")
    ax.set_ylabel("index build time [ms]")
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
    ax.legend(ncol=2)
    save(fig, "euclidean_kitti_build")

    # (b) mean query time vs N
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for lib in libs + ["brute_force"]:
        x, y = series(rows, "kitti", lib, "N", "query_us")
        if not x:
            continue
        lab, col, mk, ls = STYLE[lib]
        ax.loglog(x, y, ls=ls, marker=mk, ms=3.5, color=col, label=lab)
    ax.set_xlabel("dataset size $N$ (KITTI LiDAR points)")
    ax.set_ylabel(r"mean query time [$\mu$s]")
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
    ax.legend(ncol=2)
    save(fig, "euclidean_kitti_query")


def fig_highdim(rows):
    libs = ["proposed", "pico_tree", "flann_kdtree", "libnabo",
            "flann_4rand", "hnsw_approx", "brute_force"]
    # (a) query time vs dimension
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for lib in libs:
        x, y = series(rows, "highdim", lib, "dim", "query_us")
        if not x:
            continue
        lab, col, mk, ls = STYLE[lib]
        ax.semilogy(x, y, ls=ls, marker=mk, ms=3.5, color=col, label=lab)
    ax.set_xlabel("feature dimension $d$")
    ax.set_ylabel(r"mean query time [$\mu$s]")
    ax.set_xticks([32, 64, 128])
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
    ax.legend(ncol=2)
    save(fig, "euclidean_highdim_query")

    # (b) recall vs dimension (exposes the approximate methods)
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for lib in libs:
        x, y = series(rows, "highdim", lib, "dim", "recall")
        if not x:
            continue
        lab, col, mk, ls = STYLE[lib]
        ax.plot(x, y, ls=ls, marker=mk, ms=3.5, color=col, label=lab)
    ax.set_xlabel("feature dimension $d$")
    ax.set_ylabel("recall vs exact NN")
    ax.set_xticks([32, 64, 128])
    ax.set_ylim(0, 1.08)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
    ax.legend(ncol=2, loc="lower left")
    save(fig, "euclidean_highdim_recall")


def _shared_legend(fig, ax, ncol):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=ncol, fontsize=7,
               bbox_to_anchor=(0.5, 1.07), frameon=False)


def fig_kitti_combined(rows):
    """One full-width row: (a) build time, (b) query time, shared legend."""
    libs = ["proposed", "pico_tree", "flann_kdtree", "pcl_kdtree",
            "libnabo", "libkdtree"]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(7.0, 1.95))
    for lib in libs:
        x, y = series(rows, "kitti", lib, "N", "build_ms")
        if x:
            lab, col, mk, ls = STYLE[lib]
            a0.loglog(x, y, ls=ls, marker=mk, ms=3, color=col, label=lab)
    for lib in libs + ["brute_force"]:
        x, y = series(rows, "kitti", lib, "N", "query_us")
        if x:
            lab, col, mk, ls = STYLE[lib]
            a1.loglog(x, y, ls=ls, marker=mk, ms=3, color=col, label=lab)
    a0.set_xlabel("dataset size $N$")
    a0.set_ylabel("build time [ms]")
    a0.set_title("(a) index build", fontsize=8)
    a1.set_xlabel("dataset size $N$")
    a1.set_ylabel(r"query time [$\mu$s]")
    a1.set_title("(b) exact $1$-NN query", fontsize=8)
    for ax in (a0, a1):
        ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
    _shared_legend(fig, a1, 4)
    fig.subplots_adjust(top=0.68)
    save(fig, "euclidean_kitti")


def fig_highdim_combined(rows):
    """One full-width row: (a) query time, (b) recall, shared legend."""
    libs = ["proposed", "pico_tree", "flann_kdtree", "libnabo",
            "flann_4rand", "hnsw_approx", "brute_force"]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(7.0, 1.95))
    for lib in libs:
        x, y = series(rows, "highdim", lib, "dim", "query_us")
        if x:
            lab, col, mk, ls = STYLE[lib]
            a0.semilogy(x, y, ls=ls, marker=mk, ms=3, color=col, label=lab)
    for lib in libs:
        x, y = series(rows, "highdim", lib, "dim", "recall")
        if x:
            lab, col, mk, ls = STYLE[lib]
            a1.plot(x, y, ls=ls, marker=mk, ms=3, color=col, label=lab)
    a0.set_xlabel("feature dimension $d$")
    a0.set_ylabel(r"query time [$\mu$s]")
    a0.set_title("(a) query time", fontsize=8)
    a0.set_xticks([32, 64, 128])
    a1.set_xlabel("feature dimension $d$")
    a1.set_ylabel("recall vs exact NN")
    a1.set_title("(b) recall", fontsize=8)
    a1.set_xticks([32, 64, 128])
    a1.set_ylim(0, 1.08)
    for ax in (a0, a1):
        ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
    _shared_legend(fig, a0, 4)
    fig.subplots_adjust(top=0.68)
    save(fig, "euclidean_highdim")


def fig_euclidean_all(rows):
    """Single full-width row with the three informative Euclidean panels
    (page budget): (a) KITTI build, (b) KITTI query, (c) high-dim query;
    one shared legend covering the union of libraries. Recall is reported
    in the caption (1.0 for all exact trees, ~0.99 for the approximate
    methods at d=128)."""
    kitti_libs = ["proposed", "pico_tree", "flann_kdtree", "pcl_kdtree",
                  "libnabo", "libkdtree"]
    hd_libs = ["proposed", "pico_tree", "flann_kdtree", "libnabo",
               "flann_4rand", "hnsw_approx", "brute_force"]
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(7.05, 2.15))
    fig.set_layout_engine("none")
    for lib in kitti_libs:
        x, y = series(rows, "kitti", lib, "N", "build_ms")
        if x:
            lab, col, mk, ls = STYLE[lib]
            a0.loglog(x, y, ls=ls, marker=mk, ms=2.6, color=col, label=lab)
    for lib in kitti_libs + ["brute_force"]:
        x, y = series(rows, "kitti", lib, "N", "query_us")
        if x:
            lab, col, mk, ls = STYLE[lib]
            a1.loglog(x, y, ls=ls, marker=mk, ms=2.6, color=col)
    for lib in hd_libs:
        x, y = series(rows, "highdim", lib, "dim", "query_us")
        if x:
            lab, col, mk, ls = STYLE[lib]
            a2.semilogy(x, y, ls=ls, marker=mk, ms=2.6, color=col,
                        label=lab if lib not in kitti_libs else None)
    a0.set_xlabel("dataset size $N$\n(a) KITTI: index build")
    a0.set_ylabel("build [ms]")
    a1.set_xlabel("dataset size $N$\n(b) KITTI: exact $1$-NN query")
    a1.set_ylabel(r"query [$\mu$s]")
    a2.set_xlabel("feature dimension $d$\n(c) high-dim: $1$-NN query")
    a2.set_ylabel(r"query [$\mu$s]")
    a2.set_xticks([32, 64, 128])
    for ax in (a0, a1, a2):
        ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
        ax.tick_params(labelsize=7)
    h0, l0 = a0.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    fig.legend(h0 + h2, l0 + l2, loc="upper center", ncol=5, fontsize=7,
               bbox_to_anchor=(0.5, 1.00), frameon=False)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.30,
                        top=0.80, wspace=0.42)
    save(fig, "euclidean_all")


def main():
    rows = load()
    fig_kitti_combined(rows)
    fig_highdim_combined(rows)
    fig_euclidean_all(rows)
    print("Figures written to", PAPER_FIGS)


if __name__ == "__main__":
    main()
