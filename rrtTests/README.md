# rrtTests: RRT\* growing-tree NN on SE(2)

Benchmarks the **incremental manifold KD-tree** in the setting it is designed
for: a sampling-based motion planner (RRT / RRT\*) whose tree of explored
configurations only grows, by thousands of poses, with nearest / near-set
queries dominating runtime. The configuration space is `SE(2) = R^2 x SO(2)`
(x, y, heading), so the correct metric is geodesic and the heading wraps at
`+/-pi`.

On the identical RRT\* node sequence (same RNG, same steering) we compare:

| backend       | structure                                              | metric                |
|---------------|--------------------------------------------------------|-----------------------|
| `incremental` | `KDTreeSingleIndexIncrementalAdaptor` + `Manifold_Adaptor<SE2>` | exact SE(2) geodesic  |
| `rebuild`     | `KDTreeSingleIndexAdaptor` rebuilt every iteration     | exact SE(2) geodesic  |
| `brute`       | linear scan                                            | exact SE(2) geodesic  |
| `naive-eucl`  | incremental tree, plain `L2_Simple` on raw (x,y,theta) | Euclidean (wrong)     |

Each iteration: sample a configuration, find its nearest node (k-NN, k=1), steer
a new node toward it, query the near-set (radius), and add the new node.

## Build & run

```bash
cmake -S . -B build -DNANOFLANN_INCLUDE_DIR=$HOME/code/nanoflann/include
cmake --build build -j
./build/bench_rrt_se2 20000          # iterations (default 20000)
```

CSV columns: `backend,N,nearest_ms,near_ms,update_ms,total_ms,recall`
(`*_ms` are cumulative over the whole run; `recall` is the fraction of
iterations whose reported nearest equals the geodesic brute-force nearest).

## Results (single thread; representative run)

N = 20,001 nodes:

| backend       | nearest | near  | update   | total    | recall |
|---------------|---------|-------|----------|----------|--------|
| brute         | 743 ms  | 0.5   | 0        | 745 ms   | 1.0000 |
| **incremental** | 18 ms | 151   | **11 ms**| **182 ms** | **1.0000** |
| rebuild       | 16 ms   | 124   | 19696 ms | 19844 ms | 1.0000 |
| naive-eucl    | 14 ms   | 119   | 11 ms    | 145 ms   | 0.8216 |

Takeaways:

1. **Incremental vs rebuild-each-iteration.** The incremental index amortizes
   its self-balancing rebuilds to ~11 ms of total update cost over the whole
   run, versus ~20 s for rebuilding a static tree each iteration
   (~**100x** faster end-to-end), while keeping single-tree query speed. This
   is the cost the incremental index removes from a growing planner tree.
2. **Exactness vs the Euclidean workaround.** Treating the heading as a plain
   Euclidean coordinate is fast but returns the *wrong* nearest on ~18 % of
   queries (recall 0.82; worse for sparser trees, e.g. 0.63 at N=3000) because
   it ignores the `+/-pi` wrap. The manifold metric is exact (recall 1.0) at
   the same query speed.

So the incremental manifold KD-tree gives RRT\* both the right metric and
rebuild-free growth in one structure. (This experiment supports the paper's
motion-planning motivation; it is kept here rather than in the page-limited
manuscript.)
