#!/bin/bash
# Correctness harness for the concurrent KD-tree builder: serial-vs-concurrent
# tree equality, the same for the dynamic index, the advertised thread bound,
# and an exception thrown by user code mid-build (under ASAN).
#
# Usage: run_checks.sh <baseline-include-dir> <candidate-include-dir>
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE="${1:-$HOME/ros2_ws/src/nanoflann/include}"
CAND="${2:-$BASE}"

cmake -S "$HERE" -B "$HERE/build" -DCMAKE_BUILD_TYPE=Release \
      -DNANOFLANN_BASELINE_INCLUDE_DIR="$BASE" \
      -DNANOFLANN_CANDIDATE_INCLUDE_DIR="$CAND"
cmake --build "$HERE/build" -j"$(nproc)"

for which in baseline candidate; do
    echo
    echo "################ $which ################"
    "$HERE/build/verify_mt_build_$which" | grep -vE '^MEM '
    echo
    "$HERE/build/verify_dyn_mt_build_$which"
done

if [ -x "$HERE/build/peak_threads" ]; then
    echo
    echo "################ concurrency bound (candidate) ################"
    "$HERE/build/peak_threads"
fi

echo
echo "################ per-task pool memory (candidate) ################"
"$HERE/build/verify_mt_build_candidate" | grep -E '^MEM '

# The interesting failure modes are memory-safety ones, so rebuild the two
# harnesses that can expose them under sanitizers. ASLR has to go for TSAN.
echo
echo "################ sanitizers ################"
mkdir -p "$HERE/build/san"
for which in baseline candidate; do
    inc="$BASE"
    [ "$which" = candidate ] && inc="$CAND"
    g++ -std=c++17 -O1 -g -fsanitize=address,undefined -I "$inc" \
        "$HERE/verify_mt_build.cpp" -o "$HERE/build/san/asan_$which" -pthread
    g++ -std=c++17 -O1 -g -fsanitize=thread -I "$inc" \
        "$HERE/verify_mt_build.cpp" -o "$HERE/build/san/tsan_$which" -pthread
    g++ -std=c++17 -O1 -g -fsanitize=address -I "$inc" \
        "$HERE/exc_safety.cpp" -o "$HERE/build/san/exc_$which" -pthread

    echo "--- $which: ASAN+UBSAN ---"
    ASAN_OPTIONS=detect_leaks=1 "$HERE/build/san/asan_$which" | tail -1
    echo "--- $which: TSAN ---"
    setarch -R "$HERE/build/san/tsan_$which" | tail -1
    echo "--- $which: exception mid-build, ASAN (3 runs) ---"
    for i in 1 2 3; do
        if out=$("$HERE/build/san/exc_$which" 3000000 2>&1); then
            echo "  run $i: $(echo "$out" | grep -m1 'caught:')"
        else
            echo "  run $i: $(echo "$out" | grep -m1 -oE 'ERROR: AddressSanitizer: [a-z-]+')"
        fi
    done
done
