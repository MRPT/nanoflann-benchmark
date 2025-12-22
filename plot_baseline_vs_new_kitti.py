#!/bin/env python3

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Collect all files
files = glob.glob("stats_kitti_*.txt")

# Parse filenames into categories
data = {}
for f in files:
    base = os.path.basename(f)
    parts = base.replace(".txt", "").split("_")
    # Filename pattern: stats_kitti_00[_new]_T*
    if "new" in parts:
        method = "new"
        thread = parts[-1]  # e.g. T0
    else:
        method = "baseline"
        thread = parts[-1]

    # Load data and compute statistics
    df = pd.read_csv(
        f,
        delim_whitespace=True,
        names=["NUM_POINTS", "BUILD", "QUERY"]
    )

    build_mean = df["BUILD"].mean()
    build_p10 = df["BUILD"].quantile(0.10)
    build_p90 = df["BUILD"].quantile(0.90)

    query_mean = df["QUERY"].mean()
    query_p10 = df["QUERY"].quantile(0.10)
    query_p90 = df["QUERY"].quantile(0.90)

    data[(thread, method)] = {
        "BUILD_MEAN": build_mean,
        "BUILD_ERR_LOW": build_mean - build_p10,
        "BUILD_ERR_HIGH": build_p90 - build_mean,
        "QUERY_MEAN": query_mean,
        "QUERY_ERR_LOW": query_mean - query_p10,
        "QUERY_ERR_HIGH": query_p90 - query_mean,
    }

# Organize into a DataFrame
threads = sorted(set(t for (t, m) in data.keys()), key=lambda x: int(x[1:]))

records = []
for thread in threads:
    for method in ["baseline", "new"]:
        stats = data[(thread, method)]
        records.append({
            "Thread": int(thread[1:]),
            "Method": method,
            **stats,
        })

df = pd.DataFrame(records)

# Plot build times with 10–90 percentile error bars
plt.figure(figsize=(8, 5))
for method in ["baseline", "new"]:
    subset = df[df["Method"] == method]
    offset = -0.2 if method == "baseline" else 0.2

    yerr = [
        subset["BUILD_ERR_LOW"],
        subset["BUILD_ERR_HIGH"],
    ]

    plt.bar(
        subset["Thread"] + offset,
        subset["BUILD_MEAN"],
        width=0.4,
        yerr=yerr,
        capsize=5,
        label=method.capitalize(),
    )

plt.xlabel("Number of Threads")
plt.ylabel("Average Build Time")
plt.title("Baseline vs New: Build Time (10–90 Percentile)")
plt.legend()
plt.xticks(df["Thread"].unique())
plt.tight_layout()
plt.show()

# Plot query times with 10–90 percentile error bars
plt.figure(figsize=(8, 5))
for method in ["baseline", "new"]:
    subset = df[df["Method"] == method]
    offset = -0.2 if method == "baseline" else 0.2

    yerr = [
        subset["QUERY_ERR_LOW"],
        subset["QUERY_ERR_HIGH"],
    ]

    plt.bar(
        subset["Thread"] + offset,
        subset["QUERY_MEAN"],
        width=0.4,
        yerr=yerr,
        capsize=5,
        label=method.capitalize(),
    )

plt.xlabel("Number of Threads")
plt.ylabel("Average Query Time")
plt.title("Baseline vs New: Query Time (10–90 Percentile)")
plt.legend()
plt.xticks(df["Thread"].unique())
plt.tight_layout()
plt.show()
