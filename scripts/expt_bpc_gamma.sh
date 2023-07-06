#!/usr/bin/env bash

set -e

OUTPUT_DIR="outputs/bpc_gamma"
CONFIG_PATH=$1
DATASET=$2
C=$3

for gamma in 0.01 0.03 0.1 0.3 1 3 10 30; do
    set -x
    python -m simulate $CONFIG_PATH $DATASET bpc --output_dir $OUTPUT_DIR --gamma $gamma --c $C > /dev/null &
    set +x
    sleep 1
done
