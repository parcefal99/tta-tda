import argparse
import csv
import os
import random
import time
from datetime import datetime

import torch
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
from tda_stable import (
    update_cache,
    compute_cache_logits,
    DATASET_FULL_NAMES,
    OOD_DATASETS,
)


# ── Main adaptation loop ──────────────────────────────────────────────────────

def run_tda_unc(pos_cfg, neg_cfg, loader, clip_model, clip_weights, controls,
                margin_thresh=5.0, momentum=0.9, decay_factor=0.999,
                shot_capacity=None, unc_scale=1.0):
    """
    controls:     set of strings — any subset of {'unc', 'margin', 'momentum', 'decay'}
    unc_scale:    0 = no uncertainty effect, 1 = full scaling (alpha → 0 at max entropy)
    """
    use_unc      = 'unc'      in controls
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

            # ── uncertainty-aware alpha ───────────────────────────────────────
            # confident image → full alpha; uncertain image → reduced alpha
            if use_unc:
                confidence = max(0.0, 1.0 - unc_scale * prop_entropy)
                eff_pos_alpha = pos_params['alpha'] * confidence if pos_enabled else None
                eff_neg_alpha = neg_params['alpha'] * confidence if neg_enabled else None
            else:
                eff_pos_alpha = pos_params['alpha'] if pos_enabled else None
                eff_neg_alpha = neg_params['alpha'] if neg_enabled else None

            # ── positive cache update ─────────────────────────────────────────
            if pos_enabled:
                update_cache(
                    pos_cache, pred,
                    [image_features, loss],
                    pos_params['shot_capacity'],
                    use_momentum=use_momentum, momentum=momentum,
                    use_margin=use_margin,     margin_thresh=margin_thresh,
                    clip_logits=clip_logits,
                )

            # ── negative cache update ─────────────────────────────────────────
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

            # ── logit adjustment with uncertainty-scaled alpha ────────────────
            final_logits = clip_logits.clone()

            if pos_enabled and pos_cache:
                final_logits += compute_cache_logits(
                    image_features, pos_cache,
                    eff_pos_alpha, pos_params['beta'], clip_weights,
                    use_decay=use_decay, decay_factor=decay_factor, step=i,
                )

            if neg_enabled and neg_cache:
                final_logits -= compute_cache_logits(
                    image_features, neg_cache,
                    eff_neg_alpha, neg_params['beta'], clip_weights,
                    neg_mask_thresholds=(
                        neg_params['mask_threshold']['lower'],
                        neg_params['mask_threshold']['upper'],
                    ),
                    use_decay=use_decay, decay_factor=decay_factor, step=i,
                )

            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000)

            acc = cls_acc(final_logits, target)
            accuracies.append(acc)

            if i % 1000 == 0:
                print("---- TDA-Unc accuracy: {:.2f} | latency: {:.1f}ms/img ----\n".format(
                    sum(accuracies) / len(accuracies),
                    sum(latencies) / len(latencies)))

        final_acc   = sum(accuracies) / len(accuracies)
        avg_latency = sum(latencies) / len(latencies)
        print("---- TDA-Unc accuracy: {:.2f} | latency: {:.1f}ms/img ----\n".format(
            final_acc, avg_latency))
        return final_acc, avg_latency


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_arguments():
    p = argparse.ArgumentParser(
        description="TDA-Unc: uncertainty-aware adaptation strength.")
    p.add_argument('--config',    required=True)
    p.add_argument('--datasets',  required=True,
                   help='Datasets separated by /. E.g. A/dtd')
    p.add_argument('--backbone',  required=True, choices=['RN50', 'ViT-B/16'])
    p.add_argument('--data-root', dest='data_root', default='./dataset/')
    p.add_argument('--controls',  nargs='+', default=['unc'],
                   help="unc margin momentum decay — or 'none' for vanilla TDA.")
    p.add_argument('--unc-scale',     type=float, default=1.0,
                   help='Uncertainty scaling factor (0=disabled, 1=full effect).')
    p.add_argument('--margin-thresh', type=float, default=5.0)
    p.add_argument('--momentum',      type=float, default=0.9)
    p.add_argument('--decay-factor',  type=float, default=0.999)
    p.add_argument('--shot-capacity', type=int,   default=None)
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
    results_path = f"results/unc_{backbone_tag}_{controls_tag}_{date}.csv"
    os.makedirs("results", exist_ok=True)

    print(f"\nBackbone      : {args.backbone}")
    print(f"Controls      : {controls if controls else 'none (vanilla TDA)'}")
    print(f"unc_scale     : {args.unc_scale}")
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

        acc, latency = run_tda_unc(
            cfg['positive'], cfg['negative'],
            test_loader, clip_model, clip_weights,
            controls=controls,
            margin_thresh=args.margin_thresh,
            momentum=args.momentum,
            decay_factor=args.decay_factor,
            shot_capacity=args.shot_capacity,
            unc_scale=args.unc_scale,
        )

        write_header = not os.path.exists(results_path)
        with open(results_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'backbone', 'controls', 'unc_scale', 'margin_thresh', 'momentum',
                'decay_factor', 'shot_capacity', 'type', 'dataset',
                'accuracy', 'latency_ms',
            ])
            if write_header:
                writer.writeheader()
            writer.writerow({
                'backbone':      args.backbone,
                'controls':      controls_tag,
                'unc_scale':     args.unc_scale,
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
