#!/usr/bin/env python3
"""Plot the incremental-index KITTI sliding-window benchmark for the paper.

Reads results_revision/ablation_f80.csv, the run the numbers in Sec. V are
quoted from (same machine, same 80-frame KITTI seq-00 workload, and the run
that also carries the RSS columns behind the bytes-per-point figures), and
writes incremental_kitti.pgf into the paper figs dir plus a .png preview here.
The steady-state window is the one analyze_ablation.py reports, so the figure
and the text describe the same frames of the same run.

  top    : CDF of the per-frame map-update latency, steady state (the first
           WARMUP frames are dropped while the window is still filling).
  bottom : physically stored points per frame against the live set.

The library under test is anonymized as "incremental".
"""
import csv
import os

import numpy as np

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
PREVIEW = os.path.join(HERE, "figures")
os.makedirs(PREVIEW, exist_ok=True)

CSV = os.path.join("results_revision", "ablation_f80.csv")
WARMUP = 26          # frames dropped before the window reaches steady state
SYNC = "inc_b85_d50"  # (alpha_bal, alpha_del) = (0.85, 0.5), the paper's config

# (key, label, color, linestyle). One line style per method as well as one
# color, so the panels stay readable in grayscale and for colorblind readers.
METHODS = [
    (SYNC, "incremental (sync)", CB["blue"], "-"),
    ("inc_async", "incremental (async)", CB["sky"], "-."),
    ("ikd-Tree", "ikd-Tree", CB["orange"], "--"),
    ("forest", "Bentley--Saxe forest", CB["purple"], ":"),
    # single-threaded per-frame rebuild ("rebuild_mt" is the MT variant)
    ("rebuild", "rebuild/frame", CB["vermillion"], (0, (3, 1, 1, 1))),
]
# the forest and rebuild have no bounded-memory curve worth showing twice
MEM_METHODS = [m for m in METHODS if m[0] != "rebuild"]


def load():
    rows = []
    with open(os.path.join(HERE, CSV)) as f:
        for r in csv.DictReader(f):
            rows.append({
                "method": r["method"],
                "frame": int(r["frame"]),
                "update_ms": float(r["update_ms"]),
                "query_ms": float(r["query_ms"]),
                "live": int(r["live"]),
                "phys": int(r["phys"]),
            })
    return rows


def steady(rows, method, key):
    return np.array([r[key] for r in sorted(rows, key=lambda r: r["frame"])
                     if r["method"] == method and r["frame"] >= WARMUP])


def by_frame(rows, method, key):
    pts = sorted([r for r in rows if r["method"] == method],
                 key=lambda r: r["frame"])
    return ([r["frame"] for r in pts], [r[key] for r in pts])


def summary(rows):
    """Steady-state numbers quoted in the text, printed for cross-checking."""
    print(f"steady state = frames >= {WARMUP}")
    for key, label, _, _ in METHODS:
        u = steady(rows, key, "update_ms")
        q = steady(rows, key, "query_ms")
        if not len(u):
            continue
        print(f"  {label:22s} update med {np.median(u):8.1f} ms  "
              f"p95 {np.percentile(u, 95):8.1f} ms  "
              f"query med {np.median(q):6.2f} ms")


def figure(rows):
    fig, (ax_cdf, ax_mem) = plt.subplots(2, 1, figsize=(COL_W, 1.85))

    for key, label, color, ls in METHODS:
        v = np.sort(steady(rows, key, "update_ms"))
        if not len(v):
            continue
        ax_cdf.plot(v, np.arange(1, len(v) + 1) / len(v), ls=ls, color=color,
                    label=label)
    ax_cdf.set_xscale("log")
    ax_cdf.set_xlabel("per-frame map-update latency [ms]")
    ax_cdf.set_ylabel("CDF")
    ax_cdf.set_ylim(0, 1.02)
    ax_cdf.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
    ax_cdf.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
                  handlelength=1.8, handletextpad=0.5, labelspacing=0.3,
                  borderaxespad=0.0)

    for key, label, color, ls in MEM_METHODS:
        x, y = by_frame(rows, key, "phys")
        if not x:
            continue
        ax_mem.plot(x, np.array(y) / 1e6, ls=ls, color=color, label=label)
    x, y = by_frame(rows, SYNC, "live")
    live, = ax_mem.plot(x, np.array(y) / 1e6, ls=(0, (1, 1)), color="0.25",
                        lw=0.9, label="live points")
    ax_mem.set_xlabel("frame index")
    ax_mem.set_ylabel("points [M]")
    ax_mem.grid(True, ls=":", lw=0.4, alpha=0.6)
    # the four methods are already named in the panel above; only the live-set
    # reference is new here
    ax_mem.legend(handles=[live], loc="center left", bbox_to_anchor=(1.02, 0.5),
                  frameon=False, handlelength=1.8, handletextpad=0.5,
                  borderaxespad=0.0)

    fig.savefig(os.path.join(PAPER_FIGS, "incremental_kitti.pgf"))
    fig.savefig(os.path.join(PREVIEW, "incremental_kitti.png"), dpi=160)
    plt.close(fig)
    print("wrote incremental_kitti to", PAPER_FIGS)


def main():
    rows = load()
    summary(rows)
    figure(rows)


if __name__ == "__main__":
    main()
