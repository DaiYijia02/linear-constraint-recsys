#!/usr/bin/env bash

set -e

OUTPUT_DIR="outputs/smpca_bo"
CONFIG_PATH=$1
DATASET=$2
C=$3

for bo in 1; do
# for bo in 3 5 8 10 25 50; do
    set -x
    python -m simulate $CONFIG_PATH $DATASET smpca --c $C --output_dir $OUTPUT_DIR --b 50 --bo $bo > /dev/null &
    set +x
    sleep 1
done
