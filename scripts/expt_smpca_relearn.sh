#!/usr/bin/env bash

set -e

OUTPUT_DIR="outputs/smpca_relearn"
CONFIG_PATH=$1
DATASET=$2

# for rl in 120 90 60 30; do
for rl in 200 100 50 25; do
    for seed in $(seq 1 5); do
        set -x
        python -m simulate $CONFIG_PATH $DATASET smpca -c 0.03 --output_dir $OUTPUT_DIR -b 100 --relearn $rl --seed $seed --bo 10 > /dev/null &
        set +x
        sleep 1
    done
    sleep 7200
done
