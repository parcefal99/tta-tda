#!/bin/bash
# Uncertainty-aware weighting sweep (Stretch Goal — Part 1)
# Sweeps unc_scale, then compares unc vs all Option B controls combined.
# Usage:
#   bash ./scripts/run_unc_sweep.sh RN50
#   bash ./scripts/run_unc_sweep.sh ViT-B/16

BACKBONE="${1:-RN50}"
DATASETS="A/dtd"
ROOT="./dataset/"

cd "$(dirname "$0")/.." || exit 1

echo "=========================================="
echo "Backbone : $BACKBONE"
echo "Datasets : $DATASETS"
echo "=========================================="

echo ""
echo "--- Baseline (no controls) ---"
CUDA_VISIBLE_DEVICES=0 /workspace/miniconda3/envs/ml/bin/python tda_unc.py \
    --config configs --datasets $DATASETS --backbone "$BACKBONE" \
    --data-root $ROOT --controls none

echo ""
echo "--- SWEEP: unc_scale ---"
for SCALE in 0.3 0.5 0.7 1.0; do
    echo "  unc_scale=$SCALE"
    CUDA_VISIBLE_DEVICES=0 /workspace/miniconda3/envs/ml/bin/python tda_unc.py \
        --config configs --datasets $DATASETS --backbone "$BACKBONE" \
        --data-root $ROOT --controls unc \
        --unc-scale $SCALE
done

echo ""
echo "--- unc + all Option B controls combined ---"
CUDA_VISIBLE_DEVICES=0 /workspace/miniconda3/envs/ml/bin/python tda_unc.py \
    --config configs --datasets $DATASETS --backbone "$BACKBONE" \
    --data-root $ROOT --controls unc margin momentum decay \
    --unc-scale 0.7

echo ""
echo "=========================================="
echo "Sweep complete. Results in results/unc_*"
echo "=========================================="
