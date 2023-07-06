#!/usr/bin/env bash

set -e

OUTPUT_DIR="outputs/smpca_c_triple"
CONFIG_PATH=$1
DATASET=$2

for c in 0.01 0.03 0.1 0.3 1 10 30 100; do
    set -x
    python -m simulate $CONFIG_PATH $DATASET smpca --c $c --k 1 --output_dir $OUTPUT_DIR --b 50 --bo 10 > /dev/null &
    set +x
    sleep 1
done
