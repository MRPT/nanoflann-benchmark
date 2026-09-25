#!/usr/bin/env python3
"""Markdown table with the effect of each candidate change on its own: paired
median ratio variant / v2 (v2 = vector nodes, gather reverted, memory fixes),
from results1x/ and results10x/."""
import csv
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = ["rep", "kind", "type", "T", "irep", "frame", "N", "build_ms", "rebuild_ms",
        "knn1_us", "knn10_us", "rad_us", "used_mem", "heap", "nrad", "sink"]
CHANGES = [("v2g", "gather split coordinates (original last commit)"),
           ("v2l", "splice only task-built subtrees"),
           ("v2p", "prefetch right child")]


def load(folder, impl):
    with open(os.path.join(HERE, folder, f"static_{impl}.csv")) as f:
        return {(r["rep"], r["type"], r["T"], r["frame"]): r
                for r in (dict(zip(COLS, x)) for x in csv.reader(f) if len(x) == len(COLS))}


def ratio(folder, impl, metric, typ, threads):
    base = load(folder, "v2")
    xs = [float(r[metric]) / float(base[k][metric]) for k, r in load(folder, impl).items()
          if k in base and r["type"] == typ and int(r["T"]) in threads]
    return (st.median(xs) - 1) * 100


cells = [("build_ms", "f32", (1,), "build f32 T=1"), ("build_ms", "f32", (2, 4, 8, 16), "build f32 T>1"),
         ("build_ms", "f64", (1,), "build f64 T=1"), ("build_ms", "f64", (2, 4, 8, 16), "build f64 T>1"),
         ("knn1_us", "f32", (1, 2, 4, 8, 16), "kNN1 f32"), ("knn1_us", "f64", (1, 2, 4, 8, 16), "kNN1 f64")]
print("| change | size | " + " | ".join(c[3] for c in cells) + " |")
print("|---|---|" + "---:|" * len(cells))
for impl, label in CHANGES:
    for folder, size in (("results1x", "1x"), ("results10x", "10x")):
        vals = [ratio(folder, impl, m, t, th) for m, t, th, _ in cells]
        print(f"| {label} | {size} | " + " | ".join(f"{v:+.1f}%" for v in vals) + " |")
