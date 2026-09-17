#!/bin/bash
# Build-time / query-time benchmark of the concurrent KD-tree builder on real
# KITTI clouds. Same protocol as ../run_kitti_benchmark.sh, except the two
# nanoflann working copies are interleaved per (rep, thread-count) so CPU
# frequency drift affects both equally.
#
# Usage: run_kitti.sh <baseline-include-dir> <candidate-include-dir> [frames] [reps]
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE="${1:-$HOME/ros2_ws/src/nanoflann/include}"
CAND="${2:-$BASE}"
FRAMES="${3:-100}"
REPS="${4:-3}"
SEQ="${KITTI_SEQ:-00}"
export KITTI_SEQ="$SEQ"
: "${KITTI_BASE_DIR:?set KITTI_BASE_DIR to the KITTI odometry root}"

# The label each include dir is filed under in results/.
BASE_NAME="${BASE_NAME:-master}"
CAND_NAME="${CAND_NAME:-pr318}"

cmake -S "$HERE" -B "$HERE/build" -DCMAKE_BUILD_TYPE=Release \
      -DNANOFLANN_BASELINE_INCLUDE_DIR="$BASE" \
      -DNANOFLANN_CANDIDATE_INCLUDE_DIR="$CAND"
cmake --build "$HERE/build" -j"$(nproc)"

mkdir -p "$HERE/results"
cd "$HERE/results"
rm -f "kitti${SEQ}_${BASE_NAME}_T"*.txt "kitti${SEQ}_${CAND_NAME}_T"*.txt

THREADS=(1 2 4 8 16)
for rep in $(seq 1 "$REPS"); do
    for T in "${THREADS[@]}"; do
        for pair in "$BASE_NAME:baseline" "$CAND_NAME:candidate"; do
            name="${pair%%:*}"
            which="${pair##*:}"
            echo "rep $rep  T=$T  $name" >&2
            NANOFLANN_BENCHMARK_THREADS=$T \
                "$HERE/build/kitti_bench_${which}" "$FRAMES" 1 2>/dev/null \
                >> "kitti${SEQ}_${name}_T${T}.txt"
        done
    done
done

wc -l "kitti${SEQ}"_*_T*.txt
python3 "$HERE/plot_pr318.py"

echo
echo "### pool memory (real clouds) ###"
for pair in "$BASE_NAME:baseline" "$CAND_NAME:candidate"; do
    echo "--- ${pair%%:*} ---"
    "$HERE/build/kitti_pool_mem_${pair##*:}" 20 2>/dev/null
done

if [ -x "$HERE/build/kitti_sweep_spawncount" ]; then
    echo
    echo "### tasks spawned per build (candidate) ###"
    "$HERE/build/kitti_sweep_spawncount" 10 10 "count" 2>/dev/null
fi

echo
echo "### build time vs task-size cutoff (candidate) ###"
for b in "$HERE/build/"kitti_sweep_cutoff*; do
    [ -x "$b" ] || continue
    "$b" 10 10 "$(basename "$b" | sed 's/kitti_sweep_//')" 2>/dev/null
done
