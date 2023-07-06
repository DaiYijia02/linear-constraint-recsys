#!/usr/bin/env bash

set -e

# OUTPUT_DIR="outputs/targets_"
DATASET=$1
C=$2

for overlap in 1 2 3; do
    OUTPUT_DIR="outputs/overlap/$overlap"
    CONFIG_PATH="experiments/overlap/${overlap}.yml"
    mkdir -p $OUTPUT_DIR
    set -x
    # python -m simulate $CONFIG_PATH $DATASET bpc --c $C --output_dir $OUTPUT_DIR > /dev/null &
    # python -m simulate $CONFIG_PATH $DATASET pc --c $C --output_dir $OUTPUT_DIR > /dev/null &
    python -m simulate $CONFIG_PATH $DATASET smpca --c $C --output_dir $OUTPUT_DIR --b 50 --bo 10 > /dev/null &
    set +x
    sleep 1
done