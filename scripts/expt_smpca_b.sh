#!/usr/bin/env bash

set -e

OUTPUT_DIR="outputs/smpca_b"
CONFIG_PATH=$1
DATASET=$2
C=$3 #tv:1 kuai:0.3

for b in 1; do
# for b in 5 10 25 50 100; do
    set -x
    python -m simulate $CONFIG_PATH $DATASET smpca --c $C --output_dir $OUTPUT_DIR --b $b --bo 10 > /dev/null &
    set +x
    sleep 1
done
