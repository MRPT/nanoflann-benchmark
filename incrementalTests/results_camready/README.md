# Balance-threshold-matched comparison vs ikd-Tree

The default runs use our `(alpha_bal, alpha_del) = (0.85, 0.5)` and ikd-Tree's
FAST-LIO2 defaults `(0.7, 0.5)` (same "larger child / subtree size" criterion).
These runs check whether that difference explains the median update gap.
KITTI seq-00, `benchmark_incremental 80 60 1.5 300`, nanoflann 2.0.0, one
process per method (`BENCH_ONLY=<method>`). Summary with
`python3 ../analyze_ablation.py <csv> 26`:

| method | alpha_bal | upd median [ms] | upd p95 [ms] | query median [ms] |
|--------|-----------|-----------------|--------------|-------------------|
| `inc_b85_d50` (sync)   | 0.85 | 95.3  | 825.0  | 2.14 |
| `inc_b70_d50` (sync)   | 0.70 | 200.0 | 1966.9 | 2.09 |
| `inc_async` (MT)       | 0.85 | 71.8  | 105.3  | 2.93 |
| `inc_async_b70` (MT)   | 0.70 | 71.7  | 105.0  | 2.94 |
| `ikd-Tree`             | 0.70 | 259.8 | 338.1  | 3.36 |
| `ikdb85` (ikd-Tree)    | 0.85 | 244.2 | 336.3  | 2.70 |

The gap is not a threshold artifact: at 0.85 for both, 2.6x; at 0.70 for both,
with large rebuilds offloaded to a thread in both (ikd-Tree does so for subtrees
of >= 1500 points), 3.6x. Our synchronous tree is the one sensitive to the
threshold, since it pays the larger rebuilds on the update path.
