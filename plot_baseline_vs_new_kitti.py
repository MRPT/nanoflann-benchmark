#!/bin/env python3


import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Collect all files
files = glob.glob("stats_kitti_00*.txt")

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

    # Load data and compute averages
    df = pd.read_csv(f, delim_whitespace=True, names=[
                     "NUM_POINTS", "BUILD", "QUERY"])
    avg_build = df["BUILD"].mean()
    avg_query = df["QUERY"].mean()
    data[(thread, method)] = {"BUILD": avg_build, "QUERY": avg_query}

# Organize into a DataFrame
threads = sorted(set(t for (t, m) in data.keys()), key=lambda x: int(x[1:]))

records = []
for thread in threads:
    for method in ["baseline", "new"]:
        records.append({
            "Thread": int(thread[1:]),
            "Method": method,
            "BUILD": data[(thread, method)]["BUILD"],
            "QUERY": data[(thread, method)]["QUERY"]
        })

df = pd.DataFrame(records)

# Plot build times
plt.figure(figsize=(8, 5))
for method in ["baseline", "new"]:
    subset = df[df["Method"] == method]
    plt.bar(subset["Thread"] + (-0.2 if method == "baseline" else 0.2),
            subset["BUILD"], width=0.4, label=method.capitalize())

plt.xlabel("Number of Threads")
plt.ylabel("Average Build Time")
plt.title("Baseline vs New: Build Time")
plt.legend()
plt.xticks(df["Thread"].unique())
plt.tight_layout()
plt.show()

# Plot query times
plt.figure(figsize=(8, 5))
for method in ["baseline", "new"]:
    subset = df[df["Method"] == method]
    plt.bar(subset["Thread"] + (-0.2 if method == "baseline" else 0.2),
            subset["QUERY"], width=0.4, label=method.capitalize())

plt.xlabel("Number of Threads")
plt.ylabel("Average Query Time")
plt.title("Baseline vs New: Query Time")
plt.legend()
plt.xticks(df["Thread"].unique())
plt.tight_layout()
plt.show()
