#!/usr/bin/env python3
"""Plot the manifold head-to-head (proposed vs nigh vs MPNN vs brute force).

Reads results/manifold_libs.csv and writes .pgf into the paper figs dir and
.png previews here. The library under test is anonymized as "proposed".
"""
import csv
import os

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

# (label, color, marker, linestyle). Line style carries the method as well as
# color, so the panels stay readable in grayscale and for colorblind readers.
STYLE = {
    "proposed":    ("proposed",         CB["blue"], "o", "-"),
    "nigh":        ("nigh [Ichnowski]", CB["vermillion"], "^", "--"),
    "mpnn":        ("MPNN [Yershova]",  CB["green"], "s", "-."),
    "brute_force": ("brute force",      CB["gray"], "", ":"),
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
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    for lib in libs:
        x, y = series(rows, space, lib, "query_us")
        if not x:
            continue
        lab, col, mk, ls = STYLE[lib]
        ax.loglog(x, y, ls=ls, marker=mk, ms=3.5, color=col, label=lab)
    ax.set_xlabel("dataset size $N$ [points]")
    ax.set_ylabel(r"mean query time [$\mu$s]")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
    ax.legend()
    save(fig, name)


def fig_combined(rows):
    """One full-width row: (a) SO(3), (b) SE(3)."""
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(FULL_W, 1.75))
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
    a0.set_title(r"(a) $SO(3)$ pure rotation")
    a1.set_title(r"(b) $SE(3)$ rigid-body poses")
    for ax in (a0, a1):
        ax.set_xlabel("dataset size $N$ [points]")
        ax.set_ylabel(r"query time [$\mu$s]")
        ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
    # left panel (SO(3)) has 4 entries: two columns keep the legend at the
    # common text size without occluding the curves.
    a0.legend(loc="upper left", ncol=2, columnspacing=1.0, labelspacing=0.25,
              handlelength=1.8, handletextpad=0.4, borderpad=0.3,
              borderaxespad=0.3, framealpha=0.85)
    a1.legend(loc="upper left")
    save(fig, "manifold_libs")


def main():
    rows = load()
    fig_combined(rows)
    print("Figures written to", PAPER_FIGS)


if __name__ == "__main__":
    main()
