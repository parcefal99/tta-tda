#!/bin/bash
# Ablation: each control alone, all together, and baseline (none).
# All results land in results/stable_RN50_*.csv for easy comparison.

DATASETS="A/V/caltech101/dtd/eurosat/fgvc/food101/oxford_flowers/oxford_pets/stanford_cars/sun397/ucf101"
BACKBONE="RN50"
ROOT="./dataset/"

cd "$(dirname "$0")/.." || exit 1

for CONTROLS in "none" "margin" "momentum" "decay" "margin momentum decay"; do
    echo "=========================================="
    echo "Controls: $CONTROLS"
    echo "=========================================="
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=0 python tda_stable.py \
        --config configs \
        --datasets $DATASETS \
        --backbone $BACKBONE \
        --data-root $ROOT \
        --controls $CONTROLS
done
