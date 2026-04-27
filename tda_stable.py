import argparse
import csv
import operator
import os
import random
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from tqdm import tqdm

import clip
from utils import (
    build_test_data_loader,
    clip_classifier,
    cls_acc,
    get_clip_logits,
    get_config_file,
    get_entropy,
)

# ── Dataset metadata ──────────────────────────────────────────────────────────
DATASET_FULL_NAMES = {
    'I': 'ImageNet',       'A': 'ImageNet-A',
    'V': 'ImageNet-V2',    'R': 'ImageNet-R',
    'S': 'ImageNet-Sketch',
    'caltech101': 'Caltech101',    'dtd': 'DTD',
    'eurosat': 'EuroSAT',          'fgvc': 'FGVC-Aircraft',
    'food101': 'Food101',           'oxford_flowers': 'Flowers102',
    'oxford_pets': 'OxfordPets',   'stanford_cars': 'StanfordCars',
    'sun397': 'SUN397',             'ucf101': 'UCF101',
}
OOD_DATASETS = {'I', 'A', 'V', 'R', 'S'}

# ── Cache helpers ─────────────────────────────────────────────────────────────

def _margin(logits):
    """Top-1 minus top-2 logit value."""
    top2 = logits.topk(2, dim=1).values[0]
    return float(top2[0] - top2[1])


def update_cache(cache, pred, features_loss, shot_capacity,
                 include_prob_map=False,
                 use_momentum=False, momentum=0.9,
                 use_margin=False, margin_thresh=0.0,
                 clip_logits=None):
    """
    Drop-in replacement for the vanilla update_cache with three optional
    stability controls.

    margin_filter: skip the update entirely if the prediction margin is too low.
    momentum:      instead of a hard replace, blend the incoming feature into
                   the stored prototype.
    """
    with torch.no_grad():
        # ── margin filter ─────────────────────────────────────────────────────
        if use_margin and clip_logits is not None:
            if _margin(clip_logits) < margin_thresh:
                return  # prediction not confident enough — skip

        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]

        if pred in cache:
            if use_momentum and len(cache[pred]) > 0:
                # blend incoming feature into the first (best) stored entry
                old_feat, old_loss = cache[pred][0][0], cache[pred][0][1]
                blended_feat = momentum * old_feat + (1 - momentum) * features_loss[0]
                blended_feat = blended_feat / blended_feat.norm(dim=-1, keepdim=True)
                cache[pred][0] = [blended_feat, old_loss] + (
                    [features_loss[2]] if include_prob_map else []
                )
            else:
                if len(cache[pred]) < shot_capacity:
                    cache[pred].append(item)
                elif features_loss[1] < cache[pred][-1][1]:
                    cache[pred][-1] = item
                cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]


def compute_cache_logits(image_features, cache, alpha, beta, clip_weights,
                         neg_mask_thresholds=None,
                         use_decay=False, decay_factor=0.999, step=0):
    """
    Compute cache logits with optional exponential decay on cached affinities.

    decay: effective weight of a cache entry written t steps ago is
           decay_factor^t — old / stale entries contribute less over time.
    """
    with torch.no_grad():
        cache_keys, cache_values, cache_ages = [], [], []

        for class_index in sorted(cache.keys()):
            for idx, item in enumerate(cache[class_index]):
                cache_keys.append(item[0])
                cache_ages.append(idx)
                if neg_mask_thresholds:
                    cache_values.append(item[2])
                else:
                    cache_values.append(class_index)

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)

        if neg_mask_thresholds:
            cache_values = torch.cat(cache_values, dim=0)
            cache_values = (
                ((cache_values > neg_mask_thresholds[0]) &
                 (cache_values < neg_mask_thresholds[1]))
                .type(torch.int8)
            ).cuda().half()
        else:
            cache_values = (
                F.one_hot(torch.Tensor(cache_values).to(torch.int64),
                          num_classes=clip_weights.size(1))
            ).cuda().half()

        affinity = image_features @ cache_keys  # [1, N_cached]

        if use_decay:
            # entries written earlier (higher index = added first) decay more
            ages = torch.tensor(cache_ages, dtype=torch.float32).cuda()
            decay_weights = (decay_factor ** ages).half().unsqueeze(0)
            affinity = affinity * decay_weights

        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        return alpha * cache_logits


# ── Main adaptation loop ──────────────────────────────────────────────────────

def run_tda_stable(pos_cfg, neg_cfg, loader, clip_model, clip_weights, controls,
                   margin_thresh=5.0, momentum=0.9, decay_factor=0.999, shot_capacity=None):
    """
    controls:      set of strings — any subset of {'margin', 'momentum', 'decay'}
    margin_thresh: minimum top1-top2 logit gap to allow caching
    momentum:      blend weight on old feature (higher = slower update)
    decay_factor:  per-step decay on cached affinities (< 1 = forgetting)
    shot_capacity: override config shot_capacity for positive cache (knob sweep)
    """
    use_margin   = 'margin'   in controls
    use_momentum = 'momentum' in controls
    use_decay    = 'decay'    in controls

    with torch.no_grad():
        pos_cache, neg_cache, accuracies, latencies = {}, {}, [], []

        pos_enabled = pos_cfg['enabled']
        neg_enabled = neg_cfg['enabled']

        if pos_enabled:
            pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
            if shot_capacity is not None:
                pos_params['shot_capacity'] = shot_capacity
        if neg_enabled:
            neg_params = {k: neg_cfg[k] for k in
                          ['shot_capacity', 'alpha', 'beta',
                           'entropy_threshold', 'mask_threshold']}

        for i, (images, target) in enumerate(
                tqdm(loader, desc='Processed test images: ')):

            t_start = time.perf_counter()

            image_features, clip_logits, loss, prob_map, pred = \
                get_clip_logits(images, clip_model, clip_weights)
            target       = target.cuda()
            prop_entropy = get_entropy(loss, clip_weights)

            # ── positive cache update ─────────────────────────────────────
            if pos_enabled:
                update_cache(
                    pos_cache, pred,
                    [image_features, loss],
                    pos_params['shot_capacity'],
                    use_momentum=use_momentum, momentum=momentum,
                    use_margin=use_margin,     margin_thresh=margin_thresh,
                    clip_logits=clip_logits,
                )

            # ── negative cache update ─────────────────────────────────────
            if neg_enabled:
                lo = neg_params['entropy_threshold']['lower']
                hi = neg_params['entropy_threshold']['upper']
                if lo < prop_entropy < hi:
                    update_cache(
                        neg_cache, pred,
                        [image_features, loss, prob_map],
                        neg_params['shot_capacity'],
                        include_prob_map=True,
                    )

            # ── logit adjustment ──────────────────────────────────────────
            final_logits = clip_logits.clone()

            if pos_enabled and pos_cache:
                final_logits += compute_cache_logits(
                    image_features, pos_cache,
                    pos_params['alpha'], pos_params['beta'], clip_weights,
                    use_decay=use_decay, decay_factor=decay_factor, step=i,
                )

            if neg_enabled and neg_cache:
                final_logits -= compute_cache_logits(
                    image_features, neg_cache,
                    neg_params['alpha'], neg_params['beta'], clip_weights,
                    neg_mask_thresholds=(
                        neg_params['mask_threshold']['lower'],
                        neg_params['mask_threshold']['upper'],
                    ),
                    use_decay=use_decay, decay_factor=decay_factor, step=i,
                )

            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000)  # ms

            acc = cls_acc(final_logits, target)
            accuracies.append(acc)

            if i % 1000 == 0:
                print("---- TDA-Stable accuracy: {:.2f} | latency: {:.1f}ms/img ----\n".format(
                    sum(accuracies) / len(accuracies),
                    sum(latencies) / len(latencies)))

        final_acc     = sum(accuracies) / len(accuracies)
        avg_latency   = sum(latencies) / len(latencies)
        print("---- TDA-Stable accuracy: {:.2f} | latency: {:.1f}ms/img ----\n".format(
            final_acc, avg_latency))
        return final_acc, avg_latency


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_arguments():
    p = argparse.ArgumentParser(
        description="TDA-Stable: ablation runner for Option B controls.")
    p.add_argument('--config',    required=True)
    p.add_argument('--datasets',  required=True,
                   help='Datasets separated by /. E.g. A/V/caltech101/...')
    p.add_argument('--backbone',  required=True, choices=['RN50', 'ViT-B/16'])
    p.add_argument('--data-root', dest='data_root', default='./dataset/')
    p.add_argument('--controls',  nargs='+',
                   default=['margin', 'momentum', 'decay'],
                   help="margin momentum decay — or 'none' for vanilla TDA.")
    # knob sweep args
    p.add_argument('--margin-thresh', type=float, default=5.0,
                   help='Top1-top2 logit gap threshold for margin filter (CLIP logits scaled by 100).')
    p.add_argument('--momentum',      type=float, default=0.9,
                   help='Blend weight on old cache feature (0=replace, 1=freeze).')
    p.add_argument('--decay-factor',  type=float, default=0.999,
                   help='Per-step affinity decay (1.0=no decay).')
    p.add_argument('--shot-capacity', type=int,   default=None,
                   help='Override positive cache shot capacity from config.')
    return p.parse_args()


def main():
    args     = get_arguments()
    controls = set() if args.controls == ['none'] else set(args.controls)

    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()
    random.seed(1)
    torch.manual_seed(1)

    date         = datetime.now().strftime("%b%d_%H-%M-%S")
    backbone_tag = args.backbone.replace('/', '_')
    controls_tag = '_'.join(sorted(controls)) if controls else 'baseline'
    results_path = f"results/stable_{backbone_tag}_{controls_tag}_{date}.csv"
    os.makedirs("results", exist_ok=True)

    print(f"\nBackbone      : {args.backbone}")
    print(f"Controls      : {controls if controls else 'none (vanilla TDA)'}")
    print(f"margin_thresh : {args.margin_thresh}")
    print(f"momentum      : {args.momentum}")
    print(f"decay_factor  : {args.decay_factor}")
    print(f"shot_capacity : {args.shot_capacity or 'from config'}\n")

    for dataset_name in args.datasets.split('/'):
        full_name  = DATASET_FULL_NAMES.get(dataset_name, dataset_name)
        split_type = 'OOD' if dataset_name in OOD_DATASETS else 'cross-domain'
        print(f"=== {full_name} ({split_type}) ===")

        cfg          = get_config_file(args.config, dataset_name)
        test_loader, classnames, template = build_test_data_loader(
            dataset_name, args.data_root, preprocess)
        clip_weights = clip_classifier(classnames, template, clip_model)

        acc, latency = run_tda_stable(
            cfg['positive'], cfg['negative'],
            test_loader, clip_model, clip_weights,
            controls=controls,
            margin_thresh=args.margin_thresh,
            momentum=args.momentum,
            decay_factor=args.decay_factor,
            shot_capacity=args.shot_capacity,
        )

        write_header = not os.path.exists(results_path)
        with open(results_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'backbone', 'controls', 'margin_thresh', 'momentum',
                'decay_factor', 'shot_capacity', 'type', 'dataset',
                'accuracy', 'latency_ms',
            ])
            if write_header:
                writer.writeheader()
            writer.writerow({
                'backbone':      args.backbone,
                'controls':      controls_tag,
                'margin_thresh': args.margin_thresh,
                'momentum':      args.momentum,
                'decay_factor':  args.decay_factor,
                'shot_capacity': args.shot_capacity or cfg['positive']['shot_capacity'],
                'type':          split_type,
                'dataset':       full_name,
                'accuracy':      round(acc, 2),
                'latency_ms':    round(latency, 2),
            })
        print(f"Saved → {results_path}\n")


if __name__ == "__main__":
    main()
