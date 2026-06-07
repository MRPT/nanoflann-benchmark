#!/bin/bash
# Build and run the multi-library single-tree benchmark, then plot.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
KITTI_SEQ="${1:-$HOME/datasets/kitti/sequences/00}"
NANOFLANN_INC="${NANOFLANN_INCLUDE_DIR:-$HOME/code/nanoflann/include}"

cmake -S "$HERE" -B "$HERE/build" -DCMAKE_BUILD_TYPE=Release \
      -DNANOFLANN_INCLUDE_DIR="$NANOFLANN_INC"
cmake --build "$HERE/build" --target bench -j"$(nproc)"

mkdir -p "$HERE/results"
"$HERE/build/bench" "$KITTI_SEQ" | tee "$HERE/results/multilib_bench.csv"
python3 "$HERE/plot_multilib.py"
