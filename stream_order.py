import argparse
import csv
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import clip
from tda_stable import compute_cache_logits, update_cache
from utils import (
    build_test_data_loader,
    clip_classifier,
    cls_acc,
    get_clip_logits,
    get_config_file,
    get_entropy,
)

# ── Datasets to sweep ────────────────────────────────────────────────────────
TARGET_DATASETS = [
    ('A',   'ImageNet-A', 'OOD'),
    ('dtd', 'DTD',        'cross-domain'),
]


# ── Seeded loader ─────────────────────────────────────────────────────────────

def make_seeded_loader(base_loader, seed):
    """
    Return a new DataLoader over the same dataset in a deterministic shuffled
    order derived from `seed`.  Uses torch.utils.data.Subset so no data is
    copied and the original DatasetWrapper (with its transforms) is preserved.
    """
    dataset = base_loader.dataset
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=g).tolist()
    subset = Subset(dataset, perm)
    return DataLoader(
        subset,
        batch_size=1,
        shuffle=False,           # order is already fixed by perm
        num_workers=4,
        pin_memory=True,
    )


# ── Single-run TDA (vanilla, no extra controls) ───────────────────────────────

def run_vanilla_tda(pos_cfg, neg_cfg, loader, clip_model, clip_weights):
    """Vanilla TDA — positive + negative cache, no stability controls."""
    with torch.no_grad():
        pos_cache, neg_cache, accuracies = {}, {}, []

        pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
        neg_params = {k: neg_cfg[k] for k in
                      ['shot_capacity', 'alpha', 'beta',
                       'entropy_threshold', 'mask_threshold']}

        for images, target in tqdm(loader, desc='  images', leave=False):
            image_features, clip_logits, loss, prob_map, pred = \
                get_clip_logits(images, clip_model, clip_weights)
            target       = target.cuda()
            prop_entropy = get_entropy(loss, clip_weights)

            # positive cache
            update_cache(pos_cache, pred, [image_features, loss],
                         pos_params['shot_capacity'])

            # negative cache
            lo = neg_params['entropy_threshold']['lower']
            hi = neg_params['entropy_threshold']['upper']
            if lo < prop_entropy < hi:
                update_cache(neg_cache, pred,
                             [image_features, loss, prob_map],
                             neg_params['shot_capacity'],
                             include_prob_map=True)

            final_logits = clip_logits.clone()
            if pos_cache:
                final_logits += compute_cache_logits(
                    image_features, pos_cache,
                    pos_params['alpha'], pos_params['beta'], clip_weights)
            if neg_cache:
                final_logits -= compute_cache_logits(
                    image_features, neg_cache,
                    neg_params['alpha'], neg_params['beta'], clip_weights,
                    neg_mask_thresholds=(
                        neg_params['mask_threshold']['lower'],
                        neg_params['mask_threshold']['upper'],
                    ))

            accuracies.append(cls_acc(final_logits, target))

        return sum(accuracies) / len(accuracies)


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_arguments():
    p = argparse.ArgumentParser(description='Stream-order sensitivity of TDA.')
    p.add_argument('--config',     required=True)
    p.add_argument('--backbone',   required=True, choices=['RN50', 'ViT-B/16'])
    p.add_argument('--data-root',  dest='data_root', default='./dataset/')
    p.add_argument('--seeds',      type=int, default=5,
                   help='Number of random stream orderings to evaluate.')
    return p.parse_args()


def main():
    args = get_arguments()

    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()

    date         = datetime.now().strftime("%b%d_%H-%M-%S")
    backbone_tag = args.backbone.replace('/', '_')
    results_path = f"results/stream_order_{backbone_tag}_{date}.csv"
    os.makedirs("results", exist_ok=True)

    seeds = list(range(1, args.seeds + 1))   # seeds 1 … N

    print(f"\nBackbone : {args.backbone}")
    print(f"Seeds    : {seeds}")
    print(f"Datasets : ImageNet-A, DTD\n")
    print("=" * 60)

    all_rows = []

    for dataset_key, dataset_name, split_type in TARGET_DATASETS:
        print(f"\n>>> {dataset_name}  ({split_type})")

        cfg         = get_config_file(args.config, dataset_key)
        base_loader, classnames, template = build_test_data_loader(
            dataset_key, args.data_root, preprocess)
        clip_weights = clip_classifier(classnames, template, clip_model)

        per_seed_acc = []

        for seed in seeds:
            print(f"  seed {seed} / {args.seeds} ...", end=' ', flush=True)
            t0     = time.perf_counter()
            loader = make_seeded_loader(base_loader, seed)
            acc    = run_vanilla_tda(cfg['positive'], cfg['negative'],
                                     loader, clip_model, clip_weights)
            elapsed = time.perf_counter() - t0
            per_seed_acc.append(acc)
            print(f"acc = {acc:.2f}%  ({elapsed:.0f}s)")

            row = {
                'backbone':    args.backbone,
                'dataset':     dataset_name,
                'type':        split_type,
                'seed':        seed,
                'accuracy':    round(acc, 2),
            }
            all_rows.append(row)

            # save after every seed so partial results are not lost
            write_header = not os.path.exists(results_path)
            with open(results_path, 'a', newline='') as f:
                writer = csv.DictWriter(
                    f, fieldnames=['backbone', 'dataset', 'type', 'seed', 'accuracy'])
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            print(f"  → saved to {results_path}", flush=True)

        mean_acc = np.mean(per_seed_acc)
        std_acc  = np.std(per_seed_acc, ddof=1)   # sample std
        rng_acc  = max(per_seed_acc) - min(per_seed_acc)

        print(f"\n  {dataset_name}  —  "
              f"mean={mean_acc:.2f}%  std={std_acc:.2f}%  "
              f"range={rng_acc:.2f}%  "
              f"[{min(per_seed_acc):.2f} – {max(per_seed_acc):.2f}]")
        print(f"  {'ROBUST (std < 0.3%)' if std_acc < 0.3 else 'ORDER-SENSITIVE (std >= 0.3%)'}")

        # summary row
        all_rows.append({
            'backbone':    args.backbone,
            'dataset':     dataset_name,
            'type':        split_type,
            'seed':        'MEAN',
            'accuracy':    round(float(mean_acc), 2),
        })
        all_rows.append({
            'backbone':    args.backbone,
            'dataset':     dataset_name,
            'type':        split_type,
            'seed':        'STD',
            'accuracy':    round(float(std_acc), 2),
        })
        all_rows.append({
            'backbone':    args.backbone,
            'dataset':     dataset_name,
            'type':        split_type,
            'seed':        'RANGE',
            'accuracy':    round(float(rng_acc), 2),
        })

    # write CSV
    with open(results_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['backbone', 'dataset', 'type', 'seed', 'accuracy'])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved → {results_path}")


if __name__ == '__main__':
    main()
