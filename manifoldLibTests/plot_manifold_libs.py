#!/usr/bin/env python3
"""Plot the manifold head-to-head (proposed vs nigh vs MPNN vs brute force).

Reads results/manifold_libs.csv and writes .pgf into the paper figs dir and
.png previews here. The library under test is anonymized as "proposed".
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
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.labelsize": 9,
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

STYLE = {
    "proposed":    ("proposed",     "#1f6fb2", "o", "-"),
    "nigh":        ("nigh [Ichnowski]", "#d1622b", "^", "-"),
    "mpnn":        ("MPNN [Yershova]",  "#2c8c3c", "s", "-"),
    "brute_force": ("brute force",  "0.45", "", ":"),
}


def load():
    rows = []
    with open(os.path.join(HERE, "results", "manifold_libs.csv")) as f:
        for r in csv.DictReader(f):
            r["N"] = int(r["N"])
            for c in ("build_ms", "query_us", "recall"):
                r[c] = float(r[c])
            rows.append(r)
    return rows


def series(rows, space, lib, ykey):
    pts = sorted([r for r in rows if r["space"] == space and r["library"] == lib],
                 key=lambda r: r["N"])
    return [r["N"] for r in pts], [r[ykey] for r in pts]


def save(fig, name):
    fig.savefig(os.path.join(PAPER_FIGS, name + ".pgf"))
    fig.savefig(os.path.join(PREVIEW, name + ".png"), dpi=160)
    plt.close(fig)


def fig_space(rows, space, libs, name, title):
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for lib in libs:
        x, y = series(rows, space, lib, "query_us")
        if not x:
            continue
        lab, col, mk, ls = STYLE[lib]
        ax.loglog(x, y, ls=ls, marker=mk, ms=3.5, color=col, label=lab)
    ax.set_xlabel("dataset size $N$")
    ax.set_ylabel(r"mean query time [$\mu$s]")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
    ax.legend()
    save(fig, name)


def fig_combined(rows):
    """One full-width row: (a) SO(3), (b) SE(3)."""
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(7.0, 2.0))
    for lib in ["proposed", "nigh", "mpnn", "brute_force"]:
        x, y = series(rows, "SO3", lib, "query_us")
        if x:
            lab, col, mk, ls = STYLE[lib]
            a0.loglog(x, y, ls=ls, marker=mk, ms=3, color=col, label=lab)
    for lib in ["proposed", "nigh"]:
        x, y = series(rows, "SE3", lib, "query_us")
        if x:
            lab, col, mk, ls = STYLE[lib]
            a1.loglog(x, y, ls=ls, marker=mk, ms=3, color=col, label=lab)
    a0.set_title(r"(a) $SO(3)$ pure rotation", fontsize=8)
    a1.set_title(r"(b) $SE(3)$ rigid-body poses", fontsize=8)
    for ax in (a0, a1):
        ax.set_xlabel("dataset size $N$")
        ax.set_ylabel(r"query time [$\mu$s]")
        ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
        ax.legend(fontsize=7)
    save(fig, "manifold_libs")


def main():
    rows = load()
    fig_combined(rows)
    print("Figures written to", PAPER_FIGS)


if __name__ == "__main__":
    main()
