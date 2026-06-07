# multiLibTests — single-tree build/query comparison across NN libraries

A single self-contained binary that benchmarks **index build time**, **mean
k-NN query time**, and **recall** for many C++ nearest-neighbor libraries on a
shared dataset. Used for the Euclidean baseline / library-comparison section of
the RA-L paper (the library under test is anonymized as `proposed`).

## Datasets

1. **KITTI Velodyne LiDAR** (3D), read directly from `*.bin` (no MOLA/ROS
   dependency), stacking scans to grow `N` from 5e4 to 2e6 points.
2. **Synthetic high-dimensional features** (SIFT/SURF-like Gaussian-mixture
   clusters, L2-normalized) at `d = 32, 64, 128`, `N = 1e5`, to expose the
   curse of dimensionality.

## Libraries compared

| Library        | How          | Exact? | Notes                              |
|----------------|--------------|--------|------------------------------------|
| `proposed`     | local header | yes    | nanoflann 2.0 dev tree             |
| FLANN (exact)  | system       | yes    | `KDTreeSingleIndex`                |
| FLANN (4 rand.)| system       | approx | randomized-forest, high-dim only   |
| PCL KdTreeFLANN| system       | yes    | FLANN backend; 3D only             |
| libnabo        | FetchContent | yes    | Eigen-based, ICP standard          |
| pico_tree      | FetchContent | yes    | header-only, modern                |
| fastann        | FetchContent | (lin.) | "exact" mode is a sequential scan  |
| libkdtree++    | FetchContent | yes    | header-only; 3D only here          |
| HNSW           | FetchContent | approx | graph index, high-dim only         |
| brute force    | built-in     | yes    | ground truth + baseline            |

Each competitor is optional (`-DWITH_<LIB>=OFF`) and is compiled in only if
found at configure time (`HAVE_*` macros).

## Build & run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DNANOFLANN_INCLUDE_DIR=$HOME/code/nanoflann/include
cmake --build build --target bench -j

# arg 1: KITTI sequence dir (defaults to the one in the paper machine)
./build/bench /path/to/kitti/sequences/00 > results/multilib_bench.csv

python3 plot_multilib.py   # writes .pgf into the paper figs/ + .png previews
```

Output is one CSV row per `(dataset, dim, N, library)`:
`dataset,dim,N,library,build_ms,query_us,recall`.
