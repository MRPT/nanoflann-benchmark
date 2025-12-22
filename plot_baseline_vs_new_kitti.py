#!/bin/env python3

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


def annotate_percent_change(ax, threads, baseline_mean, baseline_err_high,
                            new_mean, new_err_high, y_offset=0.03):
    """
    Annotate percent change from baseline to new above error bars.
    Returns the maximum y value used by annotations.
    """
    max_y = 0.0

    for t, b_m, b_eh, n_m, n_eh in zip(
        threads, baseline_mean, baseline_err_high, new_mean, new_err_high
    ):
        if b_m == 0:
            continue

        pct = (n_m - b_m) / b_m * 100.0

        # Top of the higher error bar
        top = max(b_m + b_eh, n_m + n_eh)
        y = top * (1 + y_offset)

        ax.text(
            t,
            y,
            f"{pct:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

        max_y = max(max_y, y)

    return max_y


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
fig, ax = plt.subplots(figsize=(8, 5))

for method in ["baseline", "new"]:
    subset = df[df["Method"] == method]
    offset = -0.2 if method == "baseline" else 0.2

    yerr = [
        subset["BUILD_ERR_LOW"],
        subset["BUILD_ERR_HIGH"],
    ]

    ax.bar(
        subset["Thread"] + offset,
        subset["BUILD_MEAN"],
        width=0.4,
        yerr=yerr,
        capsize=5,
        label=method.capitalize(),
    )

# Annotate percent change
baseline = df[df["Method"] == "baseline"].sort_values("Thread")
new = df[df["Method"] == "new"].sort_values("Thread")

max_label_y = annotate_percent_change(
    ax,
    baseline["Thread"],
    baseline["BUILD_MEAN"],
    baseline["BUILD_ERR_HIGH"],
    new["BUILD_MEAN"],
    new["BUILD_ERR_HIGH"],
)

# Ensure labels are not clipped
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin, max(ymax, max_label_y * 1.05))

ax.set_xlabel("Number of Threads")
ax.set_ylabel("Average Build Time")
ax.set_title("Baseline vs New: Build Time (10–90 Percentile)")
ax.legend()
ax.set_xticks(df["Thread"].unique())
# plt.tight_layout()
plt.show()

# Plot query times with 10–90 percentile error bars
fig, ax = plt.subplots(figsize=(8, 5))

for method in ["baseline", "new"]:
    subset = df[df["Method"] == method]
    offset = -0.2 if method == "baseline" else 0.2

    yerr = [
        subset["QUERY_ERR_LOW"],
        subset["QUERY_ERR_HIGH"],
    ]

    ax.bar(
        subset["Thread"] + offset,
        subset["QUERY_MEAN"],
        width=0.4,
        yerr=yerr,
        capsize=5,
        label=method.capitalize(),
    )

# Annotate percent change
baseline = df[df["Method"] == "baseline"].sort_values("Thread")
new = df[df["Method"] == "new"].sort_values("Thread")

max_label_y = annotate_percent_change(
    ax,
    baseline["Thread"],
    baseline["QUERY_MEAN"],
    baseline["QUERY_ERR_HIGH"],
    new["QUERY_MEAN"],
    new["QUERY_ERR_HIGH"],
)

# Ensure labels are not clipped
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin, max(ymax, max_label_y * 1.05))

ax.set_xlabel("Number of Threads")
ax.set_ylabel("Average Query Time")
ax.set_title("Baseline vs New: Query Time (10–90 Percentile)")
ax.legend()
ax.set_xticks(df["Thread"].unique())
# plt.tight_layout()
plt.show()
