#!/usr/bin/env python3
"""Summarize and plot the incremental-index benchmark CSV (per-frame stats).

Usage: analyze.py stats.csv [out_prefix] [warmup]
Produces <out_prefix>_update.png and <out_prefix>_query.png plus a text table.
"""
import sys
import csv
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False


def load(path):
    data = defaultdict(lambda: {"frame": [], "update": [], "query": [], "live": [], "phys": []})
    with open(path) as f:
        for row in csv.DictReader(f):
            d = data[row["method"]]
            d["frame"].append(int(row["frame"]))
            d["update"].append(float(row["update_ms"]))
            d["query"].append(float(row["query_ms"]))
            d["live"].append(int(row["live"]))
            d["phys"].append(int(row["phys"]))
    return data


def main():
    path = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else "bench"
    warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    data = load(path)

    order = ["forest", "rebuild", "rebuild_mt", "inc_b85_d50", "inc_b75_d50",
             "inc_b70_d30", "inc_b65_d30", "ikd-Tree"]
    methods = [m for m in order if m in data] + [m for m in data if m not in order]

    print(f"\n{'method':<14}{'upd_med':>10}{'upd_p95':>10}{'upd_mean':>10}"
          f"{'qry_med':>10}{'qry_mean':>10}{'live':>10}{'phys':>10}")
    print("-" * 94)
    rows = []
    for m in methods:
        d = data[m]
        u = np.array(d["update"][warmup:])
        q = np.array(d["query"][warmup:])
        if len(u) == 0:
            continue
        row = (m, np.median(u), np.percentile(u, 95), u.mean(),
               np.median(q), q.mean(), d["live"][-1], d["phys"][-1])
        rows.append(row)
        print(f"{m:<14}{row[1]:>10.2f}{row[2]:>10.2f}{row[3]:>10.2f}"
              f"{row[4]:>10.3f}{row[5]:>10.3f}{row[6]:>10d}{row[7]:>10d}")

    if not HAVE_PLT:
        print("\n(matplotlib not available; skipping plots)")
        return

    for key, title, fname in [("update", "Per-frame update time [ms]", f"{prefix}_update.png"),
                              ("query", "Per-frame query time [ms]", f"{prefix}_query.png")]:
        plt.figure(figsize=(9, 5))
        for m in methods:
            d = data[m]
            plt.plot(d["frame"], d[key], label=m, linewidth=1.3)
        plt.xlabel("frame")
        plt.ylabel("ms")
        plt.title(title)
        plt.legend(fontsize=8, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fname, dpi=110)
        print(f"wrote {fname}")


if __name__ == "__main__":
    main()
