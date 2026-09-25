#!/bin/bash
# 10x (accumulated) and 1x KITTI runs over every header, back to back.
set -e
source ~/ros2_ws/install/setup.bash
cd "$(dirname "$0")"
export KITTI_BASE_DIR=$HOME/datasets/kitti/
IMPLS="baseline candidate v2 v2g v2l v2p"
./run_kitti2.sh results10x 10 20 3 0.5 $IMPLS
./run_kitti2.sh results1x 1 150 3 1.0 $IMPLS
