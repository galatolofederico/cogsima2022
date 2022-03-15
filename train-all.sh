#!/bin/bash

set -e

MODELS=(
encoderencoder
vitencoder
encoderdecoder
vitdecoder
)

#dry run
for model in "${MODELS[@]}"; do
    python train.py --model "$model" --train-batches 10 --dataset ./dataset/dev.json
done


for model in "${MODELS[@]}"; do
    python train.py --model "$model" --train-batches 10000 --save
done
