#!/usr/bin/env python3
"""Paper figure for the incremental-index experiment (RA-L).

Reads k_long.csv (KITTI seq-00 sliding-window run, ~4.8 M-point map, 80
frames; see REPORT.md §3.1) and writes one two-panel single-column figure:
  (a) CDF of per-frame map-update latency (steady state),
  (b) physically stored points vs frame (memory under churn).
Outputs .pgf into the paper's figure dir and a .png preview here.

The library under test is anonymized as "proposed" (double-blind RA-L).
"""
import csv
import os
from collections import defaultdict

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
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_FIGS = os.path.normpath(
    os.path.join(HERE, "..", "..", "..", "papers", "nanoflann-paper",
                 "ieeeral", "figs"))
PREVIEW = os.path.join(HERE, "figures")
os.makedirs(PREVIEW, exist_ok=True)

WARMUP = 10  # frames skipped for the CDF (steady state), as in analyze.py

# method key -> (label, color, linestyle)
STYLE = {
    "rebuild":     ("rebuild/frame",        "#7a4fb5", ":"),
    "forest":      ("Bentley--Saxe forest", "#c4a000", "-."),
    "ikd-Tree":    ("ikd-Tree",             "#d1622b", "--"),
    "inc_b85_d50": ("incremental (sync)",   "#1f6fb2", "-"),
    "inc_async":   ("incremental (async)",  "#2c8c3c", "-"),
}


def load():
    data = defaultdict(lambda: defaultdict(list))
    with open(os.path.join(HERE, "k_long.csv")) as f:
        for r in csv.DictReader(f):
            m = r["method"]
            # k_long.csv holds two 80-frame "rebuild" blocks (single-thread,
            # then multi-thread); keep only the first (single-thread).
            if m == "rebuild" and int(r["frame"]) in [
                    x for x in data[m]["frame"]]:
                continue
            data[m]["frame"].append(int(r["frame"]))
            data[m]["update"].append(float(r["update_ms"]))
            data[m]["live"].append(int(r["live"]))
            data[m]["phys"].append(int(r["phys"]))
    return data


def main():
    data = load()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.3, 3.15))

    # (a) update-latency CDF, steady state
    for key, (label, color, ls) in STYLE.items():
        u = np.sort(np.array(data[key]["update"][WARMUP:]))
        cdf = np.arange(1, len(u) + 1) / len(u)
        ax1.plot(u, cdf, color=color, linestyle=ls, label=label, lw=1.4)
        print(f"{key:<14} med={np.median(u):7.1f}  "
              f"p95={np.percentile(u, 95):7.1f} ms")
    ax1.set_xscale("log")
    ax1.set_xlabel("per-frame map-update latency [ms]")
    ax1.set_ylabel("CDF")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", handlelength=1.8, labelspacing=0.25,
               borderpad=0.3)

    # (b) memory under churn: physically stored vs live points
    for key, (label, color, ls) in STYLE.items():
        if key == "rebuild":
            continue  # phys == live by construction
        d = data[key]
        ax2.plot(d["frame"], np.array(d["phys"]) / 1e6, color=color,
                 linestyle=ls, label=label, lw=1.4)
    d = data["inc_b85_d50"]
    ax2.plot(d["frame"], np.array(d["live"]) / 1e6, color="0.3",
             linestyle=":", label="live points", lw=1.2)
    ax2.set_xlabel("frame")
    ax2.set_ylabel("stored points [M]")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", handlelength=1.8, labelspacing=0.25,
               borderpad=0.3)

    fig.savefig(os.path.join(PAPER_FIGS, "incremental_kitti.pgf"))
    fig.savefig(os.path.join(PREVIEW, "incremental_kitti.png"), dpi=150)
    print("wrote", os.path.join(PAPER_FIGS, "incremental_kitti.pgf"))


if __name__ == "__main__":
    main()
