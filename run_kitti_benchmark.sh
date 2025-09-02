#!/usr/bin/env bash

# List of thread counts to test
THREADS=(0 1 2 4 8 10)

# Dataset sequence
SEQ="00"

# Benchmark binary
BIN="build-Release/benchmarkTool/realTests/benchmark_nanoflann_real"

# Arguments for the benchmark binary: <MAX_TIMESTEPS> <DECIMATION_COUNTS>
ARGS="1000 1"

# Loop over thread counts
for T in "${THREADS[@]}"; do
    echo "Running benchmark with ${T} threads..."
    NANOFLANN_BENCHMARK_THREADS=$T \
    KITTI_SEQ=$SEQ \
    "$BIN" $ARGS > "stats_kitti_${SEQ}_new_T${T}.txt"
done

echo "All benchmarks completed."
