#!/usr/bin/env bash

set -e

OUTPUT_DIR="outputs/targets"
# OUTPUT_DIR="outputs/targets_"
CONFIG_PATH=$1
DATASET=$2
C=$3

# for ctrl in olp; do
#     for target in 1 1.5 2 2.5 3 3.5 4 4.5 5; do
#         for init_ in zero one; do
#             for lr in 10 5 1.0 0.1 0.01 0.001 0.0001; do
#                 set -x
#                 python -m simulate $CONFIG_PATH $DATASET $ctrl --c $C --output_dir $OUTPUT_DIR --olp_lr $lr --olp_init $init_ --targets $target $target $target > /dev/null &
#                 set +x
#                 sleep 1
#             done
#         done
#         sleep 8m
#     done
# done

# for ctrl in olp; do
# for ctrl in smpca; do
#     for target in 1 1.5 2 2.5 3 3.5 4 4.5 5; do
#         set -x
#         python -m simulate $CONFIG_PATH $DATASET $ctrl --c $C --output_dir $OUTPUT_DIR --b 50 --bo 10 --targets $target $target $target > /dev/null &
#         set +x
#         sleep 1
#     done
#     sleep 5m
# done


# for target in 1 1.5 2 2.5 3 3.5 4 4.5 5; do
#     set -x
#     python -m simulate $CONFIG_PATH $DATASET base --c $C --output_dir $OUTPUT_DIR --targets $target $target $target > /dev/null &
#     set +x
#     sleep 1
# done

for target in 1 1.5 2 2.5 3 3.5 4 4.5 5; do
    set -x
    python -m simulate $CONFIG_PATH $DATASET bpc --c $C --output_dir $OUTPUT_DIR --gamma 0.1 --targets $target $target $target > /dev/null &
    set +x
    sleep 1
done