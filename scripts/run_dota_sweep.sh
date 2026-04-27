#!/bin/bash
# TDA-DOTA ablation sweep
# Compares: vanilla DOTA vs each novel extension alone vs all combined.
# Usage:
#   bash ./scripts/run_dota_sweep.sh RN50
#   bash ./scripts/run_dota_sweep.sh ViT-B/16

BACKBONE="${1:-RN50}"
DATASETS="A/dtd"
ROOT="./dataset/"

cd "$(dirname "$0")/.." || exit 1

echo "=========================================="
echo "Backbone : $BACKBONE"
echo "Datasets : $DATASETS"
echo "=========================================="

echo ""
echo "--- vanilla DOTA (no novel extensions) ---"
CUDA_VISIBLE_DEVICES=0 /workspace/miniconda3/envs/ml/bin/python tda_dota.py \
    --config configs --datasets $DATASETS --backbone "$BACKBONE" \
    --data-root $ROOT --controls none

echo ""
echo "--- + margin gate ---"
CUDA_VISIBLE_DEVICES=0 /workspace/miniconda3/envs/ml/bin/python tda_dota.py \
    --config configs --datasets $DATASETS --backbone "$BACKBONE" \
    --data-root $ROOT --controls margin

echo ""
echo "--- + reliability weighting ---"
CUDA_VISIBLE_DEVICES=0 /workspace/miniconda3/envs/ml/bin/python tda_dota.py \
    --config configs --datasets $DATASETS --backbone "$BACKBONE" \
    --data-root $ROOT --controls reliability

echo ""
echo "--- + adaptive lambda ---"
CUDA_VISIBLE_DEVICES=0 /workspace/miniconda3/envs/ml/bin/python tda_dota.py \
    --config configs --datasets $DATASETS --backbone "$BACKBONE" \
    --data-root $ROOT --controls adaptive

echo ""
echo "--- + uncertainty scaling ---"
CUDA_VISIBLE_DEVICES=0 /workspace/miniconda3/envs/ml/bin/python tda_dota.py \
    --config configs --datasets $DATASETS --backbone "$BACKBONE" \
    --data-root $ROOT --controls unc

echo ""
echo "--- all extensions combined ---"
CUDA_VISIBLE_DEVICES=0 /workspace/miniconda3/envs/ml/bin/python tda_dota.py \
    --config configs --datasets $DATASETS --backbone "$BACKBONE" \
    --data-root $ROOT --controls margin reliability adaptive unc

echo ""
echo "=========================================="
echo "Sweep complete. Results in results/dota_*"
echo "=========================================="
