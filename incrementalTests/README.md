# Incremental dynamic k-d tree benchmark

LiDAR sliding-window benchmark comparing nanoflann's dynamic indexing
strategies against each other and against ikd-Tree:

* **forest** — `KDTreeSingleIndexDynamicAdaptor` (logarithmic forest)
* **rebuild** / **rebuild_mt** — `KDTreeSingleIndexAdaptor`, cleared+rebuilt each frame
* **incremental** — `KDTreeSingleIndexIncrementalAdaptor` (single self-balancing tree),
  swept over a few `(alpha_balance, alpha_deleted)` design points
* **ikd-Tree** — HKU-MARS [ikd-Tree](https://github.com/hku-mars/ikd-Tree) (GPLv2),
  external comparison only (cloned into `../3rdparty/ikd-Tree`, **not** committed here)

The workload accumulates scans assuming constant-velocity motion, trims the map
to a cube around the sensor (`removeOutsideBox`) after each scan, then runs a
batch of KNN queries. Data source is KITTI odometry (via `mola`) when available,
otherwise synthetic uniform-random points.

This subproject uses the **local** nanoflann header (`../../nanoflann/include` by
default; override with `-DNANOFLANN_INCLUDE_DIR=...`), so it always benchmarks
your working copy.

## Build & run

```bash
# 1) Fetch the external ikd-Tree (GPLv2; kept out of this BSD repo):
./fetch_ikdtree.sh

# 2) Configure + build:
source ~/ros2_ws/install/setup.bash    # optional: enables the KITTI (mola) loader
cmake -S . -B build && cmake --build build -j

# 3) Run.  args: <frames> <keepHalf_m> <dx_m/frame> <queries/frame> <out.csv>
export KITTI_BASE_DIR=/path/to/kitti/ KITTI_SEQ=00
./build/benchmark_incremental 24 90 1.5 200 stats.csv     # large persistent map
KITTI_BASE_DIR= ./build/benchmark_incremental 30 60 1.5 300 stats.csv   # synthetic

# 4) Tables + plots:
python3 analyze.py stats.csv out_prefix <warmup_frames>
```

The full design rationale, results and conclusions are in
[`REPORT.md`](REPORT.md); the background-rebuild threading analysis is in
[`async_rebalance.md`](async_rebalance.md).

## Dependencies
- A C++17 compiler, Eigen3, PCL (`common`) and pthreads (for ikd-Tree).
- Optional: `mola_input_kitti_dataset`, `mola_yaml`, `mrpt-maps` for the KITTI loader.
- `python3` + `numpy` + `matplotlib` for `analyze.py`.
