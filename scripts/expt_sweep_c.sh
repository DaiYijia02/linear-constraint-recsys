#!/usr/bin/env bash

set -e


CONFIG_PATH=$1
DATASET=$2

# for c in 0.001 0.01 0.1 1 10 100; do
#     OUTPUT_DIR="outputs/c_sweep_dev/$c/"
#     for gamma in 0.01 0.03 0.1 0.3 1 3 10 30; do
#         set -x 
#         python -m simulate $CONFIG_PATH $DATASET lpnc --c $c $c --output_dir $OUTPUT_DIR --gamma $gamma  --dev > /dev/null &
#         # python -m simulate $CONFIG_PATH $DATASET onlinebpc --c $c $c --output_dir $OUTPUT_DIR --bpc_lr $gamma --dev > /dev/null &
#         set +x
#         sleep 1
#     done
# done

OUTPUT_DIR="output/$DATASET/oracle"
for c in 0.0001 0.001 0.01 0.1 1 10 100 1000; do
    set -x 
    python -m simulate $CONFIG_PATH $DATASET oracle --c $c $c --output_dir $OUTPUT_DIR --save_states > /dev/null &
    set +x
    sleep 1
done

# OUTPUT_DIR="output/$DATASET/bpc"
# for c in 0.0001 0.001 0.01 0.1 1 10 100 1000; do
#     set -x 
#     python -m simulate $CONFIG_PATH $DATASET bpc --c $c $c --output_dir $OUTPUT_DIR --save_states > /dev/null &
#     set +x
#     sleep 1
# done

# OUTPUT_DIR="output/$DATASET/bpc_relax"
# for c in 0.0001 0.001 0.01 0.1 1 10 100 1000; do
#     set -x 
#     python -m simulate $CONFIG_PATH $DATASET bpc_relax --c $c $c --output_dir $OUTPUT_DIR --save_states > /dev/null &
#     set +x
#     sleep 1
# done

# OUTPUT_DIR="output/$DATASET/lpnc"
# for c in 0.0001 0.001 0.01 0.1 1 10 100 1000; do
#     set -x 
#     python -m simulate $CONFIG_PATH $DATASET lpnc --c $c $c --output_dir $OUTPUT_DIR --save_states > /dev/null &
#     set +x
#     sleep 1
# done

# OUTPUT_DIR="output/$DATASET/onlinebpcnoerror"
# for c in 0.0001 0.001 0.01 0.1 1 10 100 1000; do
#     set -x 
#     python -m simulate $CONFIG_PATH $DATASET onlinebpcnoerror --c $c $c --output_dir $OUTPUT_DIR --save_states > /dev/null &
#     set +x
#     sleep 1
# done

# OUTPUT_DIR="output/$DATASET/onlinebpctest"
# for c in 0.0001 0.001 0.01 0.1 1 10 100 1000; do
#     set -x 
#     python -m simulate $CONFIG_PATH $DATASET onlinebpctest --c $c $c --output_dir $OUTPUT_DIR --save_states > /dev/null &
#     set +x
#     sleep 1
# done


# for i in "0.001 0.01" "0.01 0.01"  "0.1 0.01" "1 0.01" "10 0.01" "100 0.01"; do
#     set -- $i
#     OUTPUT_DIR="outputs/c_sweep/$1/"
#     echo $1 $2
#     python -m simulate $CONFIG_PATH lastfm lpnc --c $1 $1 --output_dir $OUTPUT_DIR --gamma $2  > /dev/null &
#     sleep 1
# done
# for i in "0.001 0.01" "0.01 0.01"  "0.1 0.03" "1 0.1" "10 0.3" "100 1"; do
#     set -- $i
#     OUTPUT_DIR="outputs/c_sweep/$1/"
#     echo $1 $2
#     python -m simulate $CONFIG_PATH lastfm lpnc --c $1 $1 --output_dir $OUTPUT_DIR --gamma $2  > /dev/null &
#     sleep 1
# done

# for i in "0.001 0.1" "0.01 0.3"  "0.1 3" "1 0.1" "10 0.1" "100 0.1"; do
#     set -- $i
#     OUTPUT_DIR="outputs/c_sweep/$1/"
#     echo $1 $2
#     python -m simulate $CONFIG_PATH lastfm onlinebpc --c $1 $1 --output_dir $OUTPUT_DIR --bpc_lr $2 > /dev/null &
#     sleep 1
# done


# for i in "0.001 0.01" "0.01 0.01"  "0.1 0.01" "1 0.01" "10 0.01" "100 0.01"; do
#     set -- $i
#     OUTPUT_DIR="outputs/c_sweep/$1/"
#     echo $1 $2
#     python -m simulate $CONFIG_PATH tv_audience lpnc --c $1 $1 --output_dir $OUTPUT_DIR --gamma $2  > /dev/null &
#     sleep 1
# done

# for i in "0.001 1" "0.01 1"  "0.1 1" "1 1" "10 1" "100 1"; do
#     set -- $i
#     OUTPUT_DIR="outputs/c_sweep/$1/"
#     echo $1 $2
#     python -m simulate $CONFIG_PATH tv_audience onlinebpc --c $1 $1 --output_dir $OUTPUT_DIR --bpc_lr $2 > /dev/null &
#     sleep 1
# done
