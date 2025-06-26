#!/bin/bash

# Definição dos valores que podem variar
SEEDS=(0)
# DATASETS=("R8" "R52" "OH" "MR" "TREC" "SN" "AG" "SST2" "OVER")
DATASETS=("MR")
EPOCHS=(10)
BATCH_SIZES=(16)
ID_CUDA=0
TAGS="test machine1"
script="benchmark_transformers_classic.py"

for seed in "${SEEDS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for epoch in "${EPOCHS[@]}"; do
            for batch in "${BATCH_SIZES[@]}"; do
                echo "Executando: python3 $script --seed $seed --dataset $dataset --epochs $epoch --batch_size $batch --cuda $ID_CUDA --tags $TAGS"
                python3 $script --seed "$seed" --dataset "$dataset" --epochs "$epoch" --batch_size "$batch" --cuda $ID_CUDA --tags $TAGS
            done
        done
    done
done
exit 0
