# Noise-Robust Test-Time Adaptation of Vision-Language Models via Streaming Stability Controls

**Final Project Report FP17 group — Topic D**
**Abzal Nurgazy** | April 2026

## Abstract

Test-time adaptation (TTA) enables CLIP-style vision-language models to improve accuracy under distribution shift without retraining. TDA, the state-of-the-art training-free TTA method, builds dynamic positive and negative key-value caches from the test stream, but provides no quality gate on cache updates: a confidently-wrong prediction is stored and corrupts subsequent logits. We implement TDA from scratch, verify it against published results, and extend it with three independently ablatable noise-robustness controls — margin filtering, momentum blending, and affinity decay — together with an uncertainty-aware adaptation strength as a stretch goal. All four controls are evaluated on OOD (ImageNet-A) and cross-domain (DTD) benchmarks with two CLIP backbones (RN50 and ViT-B/16). Systematic ablations over confidence threshold, shot capacity, momentum, and decay factor are reported alongside per-image latency to characterise the accuracy–compute trade-off of each design choice.

> This work builds on **TDA** (Karmanov et al., CVPR 2024).
> Paper: [arxiv.org/abs/2403.18293](http://arxiv.org/abs/2403.18293) — Code: [github.com/kdiAAA/TDA](https://github.com/kdiAAA/TDA)

---

## Extensions: Streaming Stability Controls

Four independently ablatable controls implemented in `tda_stable.py`.

### Control 1 — Margin Filter
Skip the cache update if the top-1 minus top-2 logit gap is below a threshold `τ_m`. A low margin means the model is nearly equally split between two classes — such predictions are likely noisy.

```
skip update if  z_(1) - z_(2) < τ_m      (default τ_m = 5.0)
```

### Control 2 — Momentum Blending
Blend the incoming feature with the existing prototype instead of hard-replacing it:

```
f_new = (μ · f_old + (1−μ) · f_incoming) / ‖·‖₂     (default μ = 0.9)
```

### Control 3 — Affinity Decay
Down-weight older cache entries exponentially at retrieval time:

```
Â_ij = (f_i · q_j) · γ^age_j     (default γ = 0.999)
```

### Stretch Goal — Uncertainty-Aware Fusion
Scale the cache contribution at inference time by the current image's uncertainty:

```
α_eff(x) = α · max(0, 1 − s · H(x))     (default s = 1.0)
```

A confident image uses the full cache weight; a near-uniform prediction gets negligible cache support.

### Stream-Order Sensitivity (`stream_order.py`)
Evaluates how much TDA's final accuracy depends on the order images arrive. Runs vanilla TDA `N` times with different random permutations and reports mean ± std across seeds.

---

## Requirements
### Installation
```bash
git clone https://github.com/kdiAAA/TDA.git
cd TDA

conda create -n ml python=3.10
conda activate ml

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Dataset
Refer to [DATASETS.md](docs/DATASETS.md) for dataset setup.

---

## Running

### Original TDA (baseline)

```bash
# OOD benchmark
bash ./scripts/run_ood_benchmark_rn50.sh
bash ./scripts/run_ood_benchmark_vit.sh

# Cross-domain benchmark
bash ./scripts/run_cd_benchmark_rn50.sh
bash ./scripts/run_cd_benchmark_vit.sh
```

---

### Stability Controls (`tda_stable.py`)

Results saved to `results/stable_<backbone>_<controls>_<date>.csv`.

```bash
# vanilla TDA baseline (no controls)
python tda_stable.py --config configs --datasets A/V/dtd --backbone ViT-B/16 \
    --controls none

# individual controls
python tda_stable.py --config configs --datasets A/dtd --backbone ViT-B/16 \
    --controls margin

python tda_stable.py --config configs --datasets A/dtd --backbone ViT-B/16 \
    --controls momentum

python tda_stable.py --config configs --datasets A/dtd --backbone ViT-B/16 \
    --controls decay

# all three combined
python tda_stable.py --config configs --datasets A/V/dtd --backbone ViT-B/16 \
    --controls margin momentum decay

# hyperparameter sweeps
python tda_stable.py --config configs --datasets A/dtd --backbone RN50 \
    --controls margin --margin-thresh 10.0

python tda_stable.py --config configs --datasets A/dtd --backbone RN50 \
    --controls momentum --momentum 0.95

python tda_stable.py --config configs --datasets A/dtd --backbone RN50 \
    --controls decay --decay-factor 0.995

python tda_stable.py --config configs --datasets A/dtd --backbone RN50 \
    --controls none --shot-capacity 5
```

#### Arguments

| Argument | Default | Description |
|---|---|---|
| `--controls` | `margin momentum decay` | Controls to enable, or `none` for vanilla TDA |
| `--margin-thresh` | `5.0` | Top-1 minus top-2 logit gap threshold |
| `--momentum` | `0.9` | Blend weight on existing cache prototype |
| `--decay-factor` | `0.999` | Per-step affinity decay (1.0 = no decay) |
| `--shot-capacity` | from config | Override positive cache shot capacity |
| `--backbone` | — | `RN50` or `ViT-B/16` |
| `--datasets` | — | `/`-separated: `I A V R S caltech101 dtd eurosat fgvc food101 oxford_flowers oxford_pets stanford_cars sun397 ucf101` |

---

### Stream-Order Sensitivity (`stream_order.py`)

```bash
# ViT-B/16, 3 seeds (~1 hour)
python stream_order.py --config configs --backbone ViT-B/16 \
    --data-root ./dataset/ --seeds 3

# RN50, 3 seeds (~1 hour)
python stream_order.py --config configs --backbone RN50 \
    --data-root ./dataset/ --seeds 3
```

Results saved after each seed to `results/stream_order_<backbone>_<date>.csv`.

---

## Results

### Reproduction + Control Ablations

#### RN50

| Control | IN-A | DTD |
|---|:---:|:---:|
| TDA baseline (reproduced) | 30.98 | 44.09 |
| + Margin | 30.81 | 40.37 |
| + Momentum | 30.85 | 43.62 |
| + Decay | 30.98 | 44.03 |
| All three | 30.91 | 42.20 |

#### ViT-B/16

| Control | IN-A | DTD |
|---|:---:|:---:|
| TDA baseline (reproduced) | 60.43 | **46.75** |
| + Margin | **60.45** | 44.44 |
| + Momentum | 60.17 | 45.45 |
| + Decay | 60.43 | 46.69 |
| All three | 59.91 | 44.03 |

### Shot Capacity vs. Latency (RN50)

| k | IN-A (%) | DTD (%) | Latency (ms/img) |
|---|:---:|:---:|:---:|
| 1 | 30.85 | 43.79 | **429** |
| 2 | 30.91 | **43.85** | 664 |
| 3 | 30.98 | 43.56 | 640 |
| 5 | **31.00** | 43.85 | 720 |
| 8 | 30.84 | 42.91 | 739 |

---

## Project Report

Full details: `report/main.tex` — compile with pdfLaTeX on Overleaf (upload `main.tex` + `references.bib`).

## Acknowledgements
Builds on [Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter), [TPT](https://github.com/azshue/TPT), and [CoOp/CoCoOp](https://github.com/KaiyangZhou/CoOp).
