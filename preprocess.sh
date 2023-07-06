#!/usr/bin/env bash

set -e

echo "================"
echo "Sampling documents and queries"
python -m preprocess.data

echo "================"
echo "Generating embeddings"
python -m preprocess.generate_emb

echo "================"
echo "Training relevance model"
python -m preprocess.train_rel

echo "================"
echo "Estimating exposure w/o intervention"
python -m preprocess.estimate
