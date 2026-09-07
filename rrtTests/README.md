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
| `incremental` | `KDTreeSingleIndexIncrementalAdaptor` + `Manifold_Adaptor<SE2>` | squared product, exact |
| `rebuild`     | `KDTreeSingleIndexAdaptor` rebuilt every iteration     | squared product, exact |
| `brute`       | linear scan                                            | squared product, exact |
| `nigh`        | `nigh::Nigh` + `nigh::metric::SE2Space`, incremental insert | `\|dt\| + \|dtheta\|` (Cartesian sum), exact for *its own* metric |
| `naive-eucl`  | incremental tree, plain `L2_Simple` on raw (x,y,theta) | Euclidean (wrong)     |

**All backends grow along the same canonical node sequence** (fixed by the
brute-force reference), so every structure holds identical points at every
iteration. Without this, a backend whose metric picks a different nearest node
steers elsewhere and its tree diverges from the reference, which makes both the
timings and the recall incomparable.

**nigh solves a different problem.** Its `SE2Space` is a `CartesianSpace` whose
distance is the *unsquared* sum of the factor distances, `||dt|| + |dtheta|`,
whereas the product metric here is `||dt||^2 + dtheta^2`. Neither is a monotone
function of the other: on this workload the two pick the same nearest node for
only ~0.83 of queries (printed to stderr). Each backend is therefore scored
against the exhaustive ground truth *for its own metric*, and the timing
comparison should be read as cost on an identical workload, not as a ranking.

Each iteration: sample a configuration, find its nearest node (k-NN, k=1), steer
a new node toward it, query the near-set (radius), and add the new node.

## Build & run

```bash
cmake -S . -B build -DNANOFLANN_INCLUDE_DIR=$HOME/code/nanoflann/include
cmake --build build -j
./build/bench_rrt_se2 20000          # iterations (default 20000)
```

CSV columns: `backend,N,nearest_ms,near_ms,update_ms,total_ms,recall`
(the cross-metric agreement rate goes to stderr).

## Results at N = 20000 (Ryzen 7 2700X, GCC 13.3, -O3 -march=native)

```
backend       N      nearest_ms  near_ms  update_ms  total_ms  recall
brute         20001       748.1      0.5        0.0    1553.5  1.0000
incremental   20001        19.1    155.7       11.6     188.6  1.0000
rebuild       20001        16.1    127.3    20188.1   20339.8  1.0000
nigh          20001        14.9     89.5        5.6     111.9  1.0000  (own metric)
naive-eucl    20001        14.6    120.1        9.8     146.4  0.9434
```

Cross-metric NN agreement (product vs nigh Cartesian): 0.8345.

Reading: rebuilding a static tree every iteration costs ~108x the incremental
index, which is the point of the experiment. nigh, a structure purpose-built for
incremental planner queries, is ~1.7x faster than the incremental manifold index
on this insert-only workload (and answers a slightly different question, see
above); the incremental index's advantage is that it also supports deletion and
box trimming, which this workload never exercises. The naive Euclidean tree is
the only backend that is *wrong*: it returns a non-nearest node for ~5.7% of
queries.
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
