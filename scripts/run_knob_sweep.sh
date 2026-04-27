#!/bin/bash
# Knob sweep — 4 controls × their value ranges
# Datasets: ImageNet-A, ImageNet-V2 (OOD) + DTD, StanfordCars (cross-domain)
# Usage:
#   bash ./scripts/run_knob_sweep.sh RN50
#   bash ./scripts/run_knob_sweep.sh ViT-B/16

BACKBONE="${1:-RN50}"
DATASETS="A/dtd"
ROOT="./dataset/"

cd "$(dirname "$0")/.." || exit 1

echo "=========================================="
echo "Backbone : $BACKBONE"
echo "Datasets : $DATASETS"
echo "=========================================="

echo ""
echo "--- SWEEP 1: margin_thresh ---"
for THRESH in 5.0 10.0 20.0; do
    echo "  margin_thresh=$THRESH"
    CUDA_VISIBLE_DEVICES=0 /workspace/miniconda3/envs/ml/bin/python tda_stable.py \
        --config configs --datasets $DATASETS --backbone "$BACKBONE" \
        --data-root $ROOT --controls margin \
        --margin-thresh $THRESH
done

echo ""
echo "--- SWEEP 2: shot_capacity ---"
for CAP in 1 2 3 5 8; do
    echo "  shot_capacity=$CAP"
    CUDA_VISIBLE_DEVICES=0 /workspace/miniconda3/envs/ml/bin/python tda_stable.py \
        --config configs --datasets $DATASETS --backbone "$BACKBONE" \
        --data-root $ROOT --controls none \
        --shot-capacity $CAP
done

echo ""
echo "--- SWEEP 3: momentum ---"
for MOM in 0.0 0.5 0.7 0.9 0.95; do
    echo "  momentum=$MOM"
    CUDA_VISIBLE_DEVICES=0 /workspace/miniconda3/envs/ml/bin/python tda_stable.py \
        --config configs --datasets $DATASETS --backbone "$BACKBONE" \
        --data-root $ROOT --controls momentum \
        --momentum $MOM
done

echo ""
echo "--- SWEEP 4: decay_factor ---"
for DECAY in 1.0 0.999 0.995 0.99 0.95; do
    echo "  decay_factor=$DECAY"
    CUDA_VISIBLE_DEVICES=0 /workspace/miniconda3/envs/ml/bin/python tda_stable.py \
        --config configs --datasets $DATASETS --backbone "$BACKBONE" \
        --data-root $ROOT --controls decay \
        --decay-factor $DECAY
done

echo ""
echo "=========================================="
echo "Sweep complete. Results in results/stable_*"
echo "=========================================="
