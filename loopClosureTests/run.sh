#!/bin/bash
# Build and run the loop-closure retrieval benchmark, then plot.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
NANOFLANN_INC="${NANOFLANN_INCLUDE_DIR:-$HOME/code/nanoflann/include}"
KITTI="${KITTI_BASE_DIR:-$HOME/datasets/kitti}"

if [ ! -f "$NANOFLANN_INC/nanoflann.hpp" ]; then
  echo "nanoflann header not found at $NANOFLANN_INC"
  echo "Use nanoflann >= 2.0.0 (https://github.com/jlblancoc/nanoflann)"
  exit 1
fi
if [ ! -f "$KITTI/poses/00-tum.txt" ]; then
  echo "KITTI ground-truth poses not found under $KITTI/poses"
  exit 1
fi

cmake -S "$HERE" -B "$HERE/build" -DCMAKE_BUILD_TYPE=Release \
      -DNANOFLANN_INCLUDE_DIR="$NANOFLANN_INC"
cmake --build "$HERE/build" --target bench_loopclosure -j"$(nproc)"

mkdir -p "$HERE/results"
# Sequences with revisits / loop closures: 00, 02, 05, 06, 08.
(cd "$HERE" && ./build/bench_loopclosure "$KITTI/poses" 00 02 05 06 08 \
  | tee results/loopclosure.csv)
python3 "$HERE/plot_loopclosure.py"
