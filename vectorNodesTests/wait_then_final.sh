#!/bin/bash
# Waits a fixed delay, then until the machine has been idle for 10 minutes,
# then runs the final benchmarks (static 10x and 1x, then dynamic).
#
# Usage: wait_then_final.sh <delay_seconds>
cd "$(dirname "$0")"
LOG="$PWD/wait_then_final.log"
rm -f final.done finaldyn.done
echo "$(date) waiting $1 s" > "$LOG"
sleep "$1"

IGNORE='^top$|nxnode|Xorg|gnome|chrome|claude|pulseaudio|pipewire'
busy()
{
    top -bn2 -d 2 -w 200 | awk -v ign="$IGNORE" \
        '/^top -/{n++} n==2 && $1 ~ /^[0-9]+$/ && $9+0 > 20 && $12 !~ ign {print $9"% "$12}'
}
quiet=0
while [ "$quiet" -lt 20 ]; do  # 20 checks x 30 s = 10 min
    load=$(cut -d' ' -f1 /proc/loadavg)
    b=$(busy)
    if [ -z "$b" ] && awk -v l="$load" 'BEGIN{exit !(l < 1.0)}'; then
        quiet=$((quiet + 1))
    else
        [ "$quiet" -gt 0 ] && echo "$(date +%T) not idle (load $load): $b" >> "$LOG"
        quiet=0
    fi
    sleep 28
done
echo "$(date) idle, starting" >> "$LOG"

source ~/ros2_ws/install/setup.bash
export KITTI_BASE_DIR=$HOME/datasets/kitti/
IMPLS="baseline prorig final"
./run_kitti2.sh final10x 10 20 5 0.5 $IMPLS >> "$LOG" 2>&1
./run_kitti2.sh final1x 1 150 5 1.0 $IMPLS >> "$LOG" 2>&1
echo FINALDONE > final.done
./run_kitti2_dyn.sh finaldyn 1 150 3 1.0 $IMPLS >> "$LOG" 2>&1
echo DONE > finaldyn.done
echo "$(date) all done" >> "$LOG"
