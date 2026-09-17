#!/usr/bin/env python3
"""Extra node-pool memory the per-task pools cost, real KITTI clouds (20 frames).
master's concurrent build shares one pool, so its overhead is identically zero;
only PR #318 has anything to plot -> single series, no legend."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORANGE, INK, INK_MUTED = "#eb6834", "#1a1a19", "#6b6a63"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "figure.dpi": 150})

threads = [1, 2, 4, 8, 16]
serial = 2096654
pr318 = [2096654, 2100192, 2138331, 2213287, 2336132]

over_kib = np.array([(v - serial) / 1024.0 for v in pr318])
over_pct = np.array([(v - serial) / serial * 100 for v in pr318])

fig, ax = plt.subplots(figsize=(9, 4.4))
x = np.arange(len(threads))
ax.bar(x, over_kib, width=0.5, color=ORANGE)
for xi, k, p in zip(x, over_kib, over_pct):
    ax.text(xi, k + max(over_kib) * 0.035, f"+{k:.0f} KiB\n(+{p:.1f}%)",
            ha="center", va="bottom", fontsize=10, color=INK_MUTED, linespacing=1.3)

ax.set_ylim(0, max(over_kib) * 1.30)
ax.set_xticks(x)
ax.set_xticklabels([str(t) for t in threads])
ax.set_xlabel("n_thread_build")
ax.set_ylabel("Extra pool bytes (KiB)")
ax.set_title("PR #318: extra node-pool memory vs. the serial build\n"
             "KITTI seq-00, ~120k pts, leaf_max_size=10, mean of 20 frames  ·  "
             "master's shared pool: +0 KiB at every thread count",
             pad=12, color=INK, fontsize=14)
ax.grid(axis="y", linestyle="--", alpha=0.3, color=INK_MUTED)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(colors=INK_MUTED)
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "figures", "pr318_pool_memory.png"), bbox_inches="tight", facecolor="white")
print("Saved: figures/pr318_pool_memory.png")
