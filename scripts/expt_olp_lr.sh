#!/usr/bin/env bash

set -e

OUTPUT_DIR="outputs/olp_lr"
CONFIG_PATH=$1
DATASET=$2
C=$3

for init_ in zero one; do
    for lr in 10.0 3.0 1.0 0.3 0.1 0.03 0.01; do
        set -x
        python -m simulate $CONFIG_PATH $DATASET olp --c $C --output_dir $OUTPUT_DIR --olp_lr $lr --olp_init $init_ > /dev/null &
        set +x
        sleep 1
    done
done
