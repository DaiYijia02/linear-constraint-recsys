#!/usr/bin/env bash

set -e

OUTPUT_DIR="outputs/pc_c_triple"
CONFIG_PATH=$1
DATASET=$2

for c in 0.01 0.03 0.1 0.3 1 10 30 100; do
    for gamma in 0.1 0.3 1 3 10; do
        set -x
        python -m simulate $CONFIG_PATH $DATASET pc --c $c --k 1 --gamma $gamma --output_dir $OUTPUT_DIR > /dev/null &
        set +x
        sleep 1
    done
done
