#!/bin/bash

MODELS=(
    models/model_encoderencoder.pt
    models/model_vitencoder.pt
    models/model_encoderdecoder.pt
    models/model_vitdecoder.pt
    knn
)
POINTS=(10 25 50 100 150 200 250 300)

for model in "${MODELS[@]}"; do
    for points in "${POINTS[@]}"; do
        python predict.py --model "$model" --points "$points" --eval-batches 1000
    done
done