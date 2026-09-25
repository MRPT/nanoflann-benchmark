#!/bin/bash
# Static-index KITTI benchmark over any set of kitti_vn_<impl> binaries,
# interleaved per repetition. "baseline" must be one of them.
#
# Only clean samples are kept: each binary starts once no foreign process is
# busy, and a run during which one shows up is discarded and repeated (see
# clean_runs.log).
#
# Usage: run_kitti2.sh <outdir> <accum> <frames> <reps> <radius_m> <impl>...
# Needs: KITTI_BASE_DIR, and the ROS 2 workspace with mola sourced.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/$1"
export KITTI_ACCUM="$2"
FRAMES="$3"
REPS="$4"
export KITTI_RADIUS="$5"
shift 5
export KITTI_SEQ="${KITTI_SEQ:-00}"
: "${KITTI_BASE_DIR:?set KITTI_BASE_DIR to the KITTI odometry root}"

LOG="$OUT/clean_runs.log"
# Interactive desktop processes come and go in short bursts: ignore them.
IGNORE='kitti_vn|^top$|nxnode|Xorg|gnome|chrome|claude|pulseaudio|pipewire'

# Prints foreign processes above 50% CPU, measured over a 2 s window.
foreign_busy()
{
    top -bn2 -d 2 -w 200 | awk -v ign="$IGNORE" \
        '/^top -/{n++} n==2 && $1 ~ /^[0-9]+$/ && $9+0 > 50 && $12 !~ ign {print $9"% "$12}'
}

wait_quiet()
{
    local quiet=0
    while [ "$quiet" -lt 3 ]; do
        if [ -z "$(foreign_busy)" ]; then quiet=$((quiet + 1)); else quiet=0; fi
        sleep 10
    done
}

mkdir -p "$OUT"
rm -f "$OUT"/static_*.csv
for rep in $(seq 1 "$REPS"); do
    for which in "$@"; do
        while true; do
            wait_quiet
            intr="$OUT/.intruders"
            : > "$intr"
            ( while true; do foreign_busy >> "$intr"; sleep 8; done ) &
            mon=$!
            "$HERE/build-cmake/kitti_vn_$which" "$FRAMES" 1 1,2,4,8,16 static 2>/dev/null \
                | sed "s/^/$rep,/" > "$OUT/.run.csv"
            kill "$mon"
            wait "$mon" 2>/dev/null || true
            if [ -s "$intr" ]; then
                echo "$(date +%T) rep $rep $which DISCARDED: $(sort -u "$intr" | tr '\n' ';')" >> "$LOG"
            else
                cat "$OUT/.run.csv" >> "$OUT/static_$which.csv"
                echo "$(date +%T) rep $rep $which ok" >> "$LOG"
                break
            fi
        done
    done
done
rm -f "$OUT/.run.csv" "$OUT/.intruders"
wc -l "$OUT"/*.csv
