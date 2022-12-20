#!/bin/bash

#MAX_NUMBER_SCANS_PER_SEQ=1000
MAX_NUMBER_SCANS_PER_SEQ=5
NUMBER_DECIMATIONS=15

RESULTS_FILE=results-nanoflann-kitti.txt

rm $RESULTS_FILE

for seq in 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 
do
    echo ""
    echo "Running for KITTI SEQ #${seq}..."
    KITTI_SEQ=$seq ../build-Release/benchmarkTool/realTests/benchmark_nanoflann_real  $MAX_NUMBER_SCANS_PER_SEQ $NUMBER_DECIMATIONS >> $RESULTS_FILE
done
