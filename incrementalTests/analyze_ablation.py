#!/usr/bin/env python3
"""Summarize the incremental ablation run: per-method update/query latency and
memory, in the form the revision's Sec. V-E table needs.

Usage: analyze_ablation.py <stats.csv> [warmup_frames]
"""
import sys
import csv
from collections import defaultdict


def pct(v, p):
    if not v:
        return float("nan")
    v = sorted(v)
    k = (len(v) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def main():
    path = sys.argv[1]
    warmup = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    upd = defaultdict(list)
    qry = defaultdict(list)
    live = defaultdict(list)
    phys = defaultdict(list)
    rss = {}
    nq = {}

    with open(path) as fp:
        for row in csv.DictReader(fp):
            m = row["method"]
            if int(row["frame"]) < warmup:
                continue
            upd[m].append(float(row["update_ms"]))
            qry[m].append(float(row["query_ms"]))
            live[m].append(int(row["live"]))
            phys[m].append(int(row["phys"]))
            if "rss_kb_peak" in row and row["rss_kb_peak"]:
                rss[m] = (int(row["rss_kb_end"]), int(row["rss_kb_peak"]))

    hdr = (f"{'method':<24}{'upd_med':>9}{'upd_p95':>9}{'qry_med':>9}"
           f"{'phys/live':>11}{'RSS_end':>10}{'RSS_peak':>10}{'B/live':>9}")
    print(hdr)
    print("-" * len(hdr))
    for m in upd:
        l = live[m][-1] if live[m] else 0
        p = phys[m][-1] if phys[m] else 0
        e, pk = rss.get(m, (0, 0))
        bpl = (e * 1024.0 / l) if l else float("nan")
        print(f"{m:<24}{pct(upd[m],50):>9.1f}{pct(upd[m],95):>9.1f}"
              f"{pct(qry[m],50):>9.2f}{(p/l if l else 0):>11.2f}"
              f"{e/1024.0:>9.0f}M{pk/1024.0:>9.0f}M{bpl:>9.0f}")
    print(f"\n(steady state: frames >= {warmup}; RSS in MiB, B/live = resident bytes per live point)")


if __name__ == "__main__":
    main()
