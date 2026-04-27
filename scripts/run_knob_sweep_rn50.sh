#!/bin/bash
# Knob sweep for 3.2 ablations — sweeps margin_thresh, shot_capacity, momentum
# Run on two representative datasets: ImageNet-A (OOD) + EuroSAT (cross-domain)
# Each sweep varies one knob at a time, others held at default.

DATASETS="A/eurosat"
BACKBONE="RN50"
ROOT="./dataset/"

cd "$(dirname "$0")/.." || exit 1

echo "=========================================="
echo "SWEEP 1: margin_thresh (margin filter)"
echo "=========================================="
for THRESH in 0.0 1.0 2.0 5.0 10.0 20.0; do
    CUDA_VISIBLE_DEVICES=0 /workspace/miniconda3/envs/ml/bin/python tda_stable.py \
        --config configs --datasets $DATASETS --backbone $BACKBONE \
        --data-root $ROOT --controls margin \
        --margin-thresh $THRESH
done

echo "=========================================="
echo "SWEEP 2: shot_capacity (cache size)"
echo "=========================================="
for CAP in 1 2 3 5 8; do
    CUDA_VISIBLE_DEVICES=0 /workspace/miniconda3/envs/ml/bin/python tda_stable.py \
        --config configs --datasets $DATASETS --backbone $BACKBONE \
        --data-root $ROOT --controls none \
        --shot-capacity $CAP
done

echo "=========================================="
echo "SWEEP 3: momentum value"
echo "=========================================="
for MOM in 0.0 0.5 0.7 0.9 0.95; do
    CUDA_VISIBLE_DEVICES=0 /workspace/miniconda3/envs/ml/bin/python tda_stable.py \
        --config configs --datasets $DATASETS --backbone $BACKBONE \
        --data-root $ROOT --controls momentum \
        --momentum $MOM
done

echo "=========================================="
echo "SWEEP 4: decay_factor"
echo "=========================================="
for DECAY in 1.0 0.999 0.995 0.99 0.95; do
    CUDA_VISIBLE_DEVICES=0 /workspace/miniconda3/envs/ml/bin/python tda_stable.py \
        --config configs --datasets $DATASETS --backbone $BACKBONE \
        --data-root $ROOT --controls decay \
        --decay-factor $DECAY
done
