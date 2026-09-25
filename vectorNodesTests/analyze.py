#!/usr/bin/env python3
"""Summarize results/{static,dynamic}_*.csv written by run_kitti.sh into
results/table.md (medians over frames and repetitions, ratios vs baseline)."""
import csv
import glob
import os
import sys
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "results")
# Baseline first, then every other static_<impl>.csv found.
_found = sorted(os.path.basename(f)[len("static_"):-len(".csv")]
                for f in glob.glob(os.path.join(RES, "static_*.csv")))
IMPLS = ["baseline"] + [i for i in _found if i != "baseline"]

S_COLS = ["rep", "kind", "type", "T", "irep", "frame", "N", "build_ms", "rebuild_ms",
          "knn1_us", "knn10_us", "rad_us", "used_mem", "heap", "nrad", "sink"]
D_COLS = ["rep", "kind", "type", "T", "irep", "mapN", "add_ms", "q_us", "heap", "sink"]


def load(kind, impl, cols):
    path = os.path.join(RES, f"{kind}_{impl}.csv")
    rows = []
    with open(path) as f:
        for r in csv.reader(f):
            if len(r) != len(cols):
                continue
            rows.append(dict(zip(cols, r)))
    return rows


def med(xs):
    return st.median(xs) if xs else float("nan")


out = []
out.append(f"Implementations: {', '.join(IMPLS)}\n")

# ---- static index ---------------------------------------------------------
static = {i: load("static", i, S_COLS) for i in IMPLS}
metrics = [("build_ms", "fresh build [ms]"), ("rebuild_ms", "in-place rebuild [ms]"),
           ("knn1_us", "kNN k=1 [µs/q]"), ("knn10_us", "kNN k=10 [µs/q]"),
           ("rad_us", "radius 1 m [µs/q]"), ("used_mem", "usedMemory() [KiB]"),
           ("heap", "heap kept by index [KiB]")]

agg = defaultdict(list)  # (impl, type, T, metric) -> values
paired = defaultdict(list)  # (impl, type, T, metric) -> per-sample ratio vs baseline
for impl, rows in static.items():
    for r in rows:
        for m, _ in metrics:
            v = float(r[m])
            if m in ("used_mem", "heap"):
                v /= 1024
            agg[(impl, r["type"], int(r["T"]), m)].append(v)

# Paired ratios: same (rep, T, type, frame) in baseline and the other impl.
base_idx = {(r["rep"], r["type"], r["T"], r["frame"]): r for r in static.get("baseline", [])}
for impl in IMPLS[1:]:
    for r in static[impl]:
        b = base_idx.get((r["rep"], r["type"], r["T"], r["frame"]))
        if not b:
            continue
        for m, _ in metrics:
            bv = float(b[m])
            if bv > 0:
                paired[(impl, r["type"], int(r["T"]), m)].append(float(r[m]) / bv)

types = sorted({k[1] for k in agg})
threads = sorted({k[2] for k in agg})
n_samples = {k: len(v) for k, v in agg.items()}

for t in types:
    out.append(f"\n## Static index, `{t}` (KITTI 00, leaf_max_size=10)\n")
    hdr = "| metric | T | " + " | ".join(IMPLS) + " | " + " | ".join(
        f"{i}/base (paired median)" for i in IMPLS[1:]) + " |"
    out.append(hdr)
    out.append("|" + "---|" * (2 + len(IMPLS) + len(IMPLS) - 1))
    for m, label in metrics:
        for T in threads:
            vals = [med(agg[(i, t, T, m)]) for i in IMPLS]
            fmt = "{:.0f}" if m in ("used_mem", "heap") else "{:.3f}"
            ratios = [med(paired[(i, t, T, m)]) for i in IMPLS[1:]]
            out.append(f"| {label} | {T} | " + " | ".join(fmt.format(v) for v in vals) + " | " +
                       " | ".join(f"{x:.3f}" for x in ratios) + " |")
    out.append("")
    some = next(iter(n_samples.values()))
    out.append(f"({some} samples per cell: frames x repetitions)\n")

# ---- dynamic index --------------------------------------------------------
if all(os.path.exists(os.path.join(RES, f"dynamic_{i}.csv")) for i in IMPLS):
    dyn = {i: load("dynamic", i, D_COLS) for i in IMPLS}
    out.append("\n## Dynamic index (map grown scan by scan, every 4th point)\n")
    out.append("| T | metric | " + " | ".join(IMPLS) + " |")
    out.append("|" + "---|" * (2 + len(IMPLS)))
    for T in sorted({int(r["T"]) for r in dyn[IMPLS[0]]}):
        for m, label, sc in [("add_ms", "total addPoints [ms]", 1),
                             ("q_us", "kNN k=1 [µs/q]", 1),
                             ("heap", "heap kept by index [MiB]", 1 / 1048576)]:
            vals = [med([float(r[m]) * sc for r in dyn[i] if int(r["T"]) == T]) for i in IMPLS]
            out.append(f"| {T} | {label} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    mapn = dyn[IMPLS[0]][0]["mapN"]
    out.append(f"\n(map size at the end: {mapn} points)\n")

text = "\n".join(out)
with open(os.path.join(RES, "table.md"), "w") as f:
    f.write(text + "\n")
print(text)
