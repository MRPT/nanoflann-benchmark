#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 (baseline|new) <MAX_FRAMES_PER_SEQUENCE>"
    exit 1
fi

METHOD_NAME=$1
MAX_LIDAR_FRAMES=$2

# List of thread counts to test
THREADS=(0 1 2 4 8)

# Dataset sequence
SEQ="00"

# Benchmark binary
BIN="build-Release/benchmarkTool/realTests/benchmark_nanoflann_real"

# Arguments for the benchmark binary: <MAX_TIMESTEPS> <DECIMATION_COUNTS>
ARGS="${MAX_LIDAR_FRAMES} 1"

# Loop over thread counts
for T in "${THREADS[@]}"; do
    echo "Running benchmark with ${T} threads..."
    NANOFLANN_BENCHMARK_THREADS=$T \
    KITTI_SEQ=$SEQ \
    "$BIN" $ARGS > "stats_kitti_${SEQ}_${METHOD_NAME}_T${T}.txt"
done

echo "All benchmarks completed."
