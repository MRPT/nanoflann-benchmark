#!/usr/bin/env python3
"""
Analyze §6.2 (middleSplit allocation) + §6.7 (else-if in computeInitialDistances)
benchmark results.

Usage:
  python3 analyze_middlesplit.py stats_middlesplit_baseline.csv stats_middlesplit_new.csv
"""

import sys
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

def load(path, label):
    df = pd.read_csv(path)
    df["version"] = label
    return df

def summarize(df, groupby_cols):
    grp = df.groupby(groupby_cols)
    out = grp.agg(
        n=("build_s", "count"),
        build_mean=("build_s", "mean"),
        build_std=("build_s", "std"),
        build_p10=("build_s", lambda x: x.quantile(0.10)),
        build_p90=("build_s", lambda x: x.quantile(0.90)),
        query_mean=("query_us", "mean"),
        query_std=("query_us", "std"),
        query_p10=("query_us", lambda x: x.quantile(0.10)),
        query_p90=("query_us", lambda x: x.quantile(0.90)),
    ).reset_index()
    return out

def welch_pvalue(a, b):
    """Two-sided Welch t-test p-value (unequal variances)."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    _, p = scipy_stats.ttest_ind(a, b, equal_var=False)
    return p

def fmt_pct(pct):
    arrow = "▲" if pct > 0 else "▼"
    return f"{arrow}{abs(pct):.1f}%"

def significance(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

baseline_path = sys.argv[1] if len(sys.argv) > 1 else "stats_middlesplit_baseline.csv"
new_path      = sys.argv[2] if len(sys.argv) > 2 else "stats_middlesplit_new.csv"

base = load(baseline_path, "baseline")
new  = load(new_path,      "new")
all_df = pd.concat([base, new], ignore_index=True)

print("=" * 80)
print("nanoflann §6.2 + §6.7 Benchmark Results")
print(f"  Baseline: {baseline_path}  ({len(base)} rows)")
print(f"  New:      {new_path}  ({len(new)} rows)")
print("=" * 80)

# Group by (source_type, leaf_max_size, k)
all_df["source_type"] = all_df["source"].apply(
    lambda s: "kitti" if s == "kitti" else s
)

groups = [("kitti", "KITTI seq-00"), ("random_10000", "Random 10k"),
          ("random_50000", "Random 50k"), ("random_120000", "Random 120k")]

for src, src_label in groups:
    sub = all_df[all_df["source"] == src]
    if sub.empty:
        continue

    print(f"\n{'─'*80}")
    print(f"  {src_label}  (n_points={sub[sub['version']=='baseline']['n_points'].median():.0f})")
    print(f"{'─'*80}")

    leaf_sizes = sorted(sub["leaf_max_size"].unique())
    k_vals     = sorted(sub["k"].unique())

    # Build time table
    print("\n  BUILD TIME (mean ± std) [seconds]")
    header = f"  {'leaf':>6}  {'k':>4}  {'baseline':>12}  {'new':>12}  {'Δ':>8}  {'p':>8}  {'sig':>4}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for ls in leaf_sizes:
        for k in k_vals:
            b = sub[(sub["version"] == "baseline") & (sub["leaf_max_size"] == ls) & (sub["k"] == k)]["build_s"]
            n = sub[(sub["version"] == "new")      & (sub["leaf_max_size"] == ls) & (sub["k"] == k)]["build_s"]
            if b.empty or n.empty: continue
            pct = (n.mean() - b.mean()) / b.mean() * 100
            p   = welch_pvalue(b.values, n.values)
            print(f"  {ls:>6}  {k:>4}  {b.mean():>10.4f}s  {n.mean():>10.4f}s  "
                  f"{fmt_pct(pct):>8}  {p:>8.4f}  {significance(p):>4}")

    # Query time table
    print("\n  QUERY TIME (mean ± std) [µs/query]")
    print(header.replace("[seconds]","[µs/query]"))
    print("  " + "-" * (len(header) - 2))

    for ls in leaf_sizes:
        for k in k_vals:
            b = sub[(sub["version"] == "baseline") & (sub["leaf_max_size"] == ls) & (sub["k"] == k)]["query_us"]
            n = sub[(sub["version"] == "new")      & (sub["leaf_max_size"] == ls) & (sub["k"] == k)]["query_us"]
            if b.empty or n.empty: continue
            pct = (n.mean() - b.mean()) / b.mean() * 100
            p   = welch_pvalue(b.values, n.values)
            print(f"  {ls:>6}  {k:>4}  {b.mean():>10.4f}µs  {n.mean():>10.4f}µs  "
                  f"{fmt_pct(pct):>8}  {p:>8.4f}  {significance(p):>4}")

print("\n")
print("Significance: *** p<0.001  ** p<0.01  * p<0.05  ns p≥0.05")
print("Δ = (new - baseline) / baseline × 100%  (▼ = improvement, ▲ = regression)")
