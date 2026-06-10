#!/bin/bash
# Build and run the C5 loop-closure benchmark on MulRan ground-truth poses.
# Appends to results/loopclosure.csv (header skipped) so KITTI + MulRan rows
# can be plotted together; run run.sh (KITTI) first for a fresh CSV.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
NANOFLANN_INC="${NANOFLANN_INCLUDE_DIR:-$HOME/code/nanoflann-manifold/include}"
MULRAN="${MULRAN_BASE_DIR:-/media/jlblanco/portable/MulRan}"
# Sequences with revisits; adjust to what is on disk.
SEQS="${MULRAN_SEQS:-KAIST01 DCC01 Riverside01}"

if [ ! -d "$MULRAN" ]; then
  echo "MulRan dataset not found at $MULRAN (is the drive mounted?)"
  exit 1
fi

cmake -S "$HERE" -B "$HERE/build" -DCMAKE_BUILD_TYPE=Release \
      -DNANOFLANN_INCLUDE_DIR="$NANOFLANN_INC"
cmake --build "$HERE/build" --target bench_loopclosure -j"$(nproc)"

mkdir -p "$HERE/results"
(cd "$HERE" && ./build/bench_loopclosure --mulran "$MULRAN" $SEQS \
  | tail -n +2 | tee -a results/loopclosure.csv)
python3 "$HERE/plot_loopclosure.py"
