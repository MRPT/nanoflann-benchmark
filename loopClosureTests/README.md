# loopClosureTests — C5 application benchmark

Loop-closure candidate retrieval over real SE(3) keyframe poses (KITTI odometry
ground truth, TUM-format files under `$KITTI_BASE_DIR/poses/`), for the paper's
claim **C5** (application-level benefit of exact manifold NN).

For every keyframe, retrieve the `k` nearest keyframes (temporal exclusion
`|i-j| > 100` frames) under the SE(3) product metric
`d^2 = ||dt||^2 + wrot^2 * chordal^2(q)` (rotation weight applied by scaling the
quaternion coordinates; no library support needed).

Methods: exhaustive geodesic scan (ground truth), the proposed manifold KD-tree
(exact), a naive Euclidean 7-D KD-tree on raw coordinates (canonicalized and
sign-continuous quaternion storage — both common, both wrong), and the
3-D-translation-tree + re-rank workaround.

Run:

```bash
./run.sh    # builds, runs seqs 00 02 05 06 08, writes results/ + figures
```

Outputs: `results/loopclosure.csv`, per-query miss dumps
`results/perquery_*.csv`, and `.pgf` figures + the LaTeX table body for the
paper (`loopclosure_recall.pgf`, `loopclosure_missmap.pgf`).

Needs the manifold-branch header worktree:
`git -C ~/code/nanoflann worktree add ~/code/nanoflann-manifold feat/manifold-topology`.
