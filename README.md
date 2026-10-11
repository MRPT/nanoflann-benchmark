# nanoflann-benchmark
Benchmarking tests, scripts and results for the nanoflann KD-tree library

See original project repo: https://github.com/jlblancoc/nanoflann

## Paper experiments

Code, raw results and plot scripts for the nanoflann 2.0 paper (IEEE RA-L 2026).
All of them need nanoflann >= 2.0.0 (`-DNANOFLANN_INCLUDE_DIR=...`, default
`~/code/nanoflann/include`). Third-party baselines are fetched at fixed versions
or commits (see each `CMakeLists.txt` and `incrementalTests/fetch_ikdtree.sh`).

| Directory | Experiment |
|-----------|------------|
| `multiLibTests/` | Euclidean library comparison (build/query, KITTI and high-dim) |
| `manifoldLibTests/` | Non-Euclidean head-to-head vs MPNN and nigh |
| `loopClosureTests/` | Loop-closure retrieval on KITTI and MulRan |
| `incrementalTests/` | Incremental index vs ikd-Tree, update latency, memory and ablation |
| `rrtTests/` | RRT* on SE(2) |

If you use these benchmarks, please cite:

```bibtex
@article{blancoclaraco2026nanoflann,
  title   = {nanoflann: A Header-Only {KD}-Tree Library for Exact Nearest-Neighbor Search on Compile-Time Product Manifolds},
  author  = {Blanco-Claraco, Jose Luis},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2026},
  note    = {Accepted for publication}
}
```
