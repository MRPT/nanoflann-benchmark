#!/bin/bash
# Build both variants, run them interleaved, verify identical checksums,
# and print the on/off time ratios.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
KITTI_SEQ="${1:-$HOME/datasets/kitti/sequences/00}"
REPS="${2:-5}"

cmake -S "$HERE" -B "$HERE/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$HERE/build" -j"$(nproc)"

mkdir -p "$HERE/results"
"$HERE/build/bench_overhead_on" "$KITTI_SEQ" "$REPS" \
    | tee "$HERE/results/overhead_on.csv"
"$HERE/build/bench_overhead_off" "$KITTI_SEQ" "$REPS" \
    | tee "$HERE/results/overhead_off.csv"

python3 "$HERE/compare.py" \
    "$HERE/results/overhead_on.csv" "$HERE/results/overhead_off.csv"
