#!/bin/bash
# Build and run the manifold head-to-head benchmark, then plot.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
NANOFLANN_INC="${NANOFLANN_INCLUDE_DIR:-$HOME/code/nanoflann-manifold/include}"

# Ensure the manifold-branch worktree exists.
if [ ! -f "$NANOFLANN_INC/nanoflann.hpp" ]; then
  echo "Manifold header not found at $NANOFLANN_INC"
  echo "Create it with: git -C \$HOME/code/nanoflann worktree add \$HOME/code/nanoflann-manifold feat/manifold-topology"
  exit 1
fi

cmake -S "$HERE" -B "$HERE/build" -DCMAKE_BUILD_TYPE=Release \
      -DNANOFLANN_INCLUDE_DIR="$NANOFLANN_INC"
cmake --build "$HERE/build" --target bench_manifold -j"$(nproc)"

mkdir -p "$HERE/results"
"$HERE/build/bench_manifold" | tee "$HERE/results/manifold_libs.csv"
python3 "$HERE/plot_manifold_libs.py"
