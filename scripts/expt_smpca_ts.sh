#!/usr/bin/env bash

set -e

OUTPUT_DIR="outputs/smpca_ts"
CONFIG_PATH=$1
DATASET=$2
C=$3
B_ONLINE=10

# for ts in 200 400 600 800; do
for ts in 200 300 400 500; do
    for seed in $(seq 1 5); do
        set -x
        python -m simulate $CONFIG_PATH $DATASET smpca --c $C --output_dir $OUTPUT_DIR --b 50 --bo $B_ONLINE --train_size $ts --seed $seed > /dev/null &
        set +x
        sleep 1
    done
    sleep 1h
done
