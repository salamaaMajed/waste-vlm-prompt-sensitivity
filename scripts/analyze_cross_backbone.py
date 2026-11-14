#!/usr/bin/env python3
"""Reconstruct paired effects from actual per-image outputs, never summaries."""
import argparse
import json
from pathlib import Path
import pandas as pd
from wastevlm.paired_analysis import align_predictions, bootstrap_family_scores, contrast_rows


def run(args):
    root = Path(args.results) / 'zero_shot' / args.dataset
    frames = {}
    reference_prompts = None
    reference_split = None
    for model in ('clip', 'siglip', 'openclip'):
        path = root / model / 'predictions.csv'
        if not path.is_file():
            raise FileNotFoundError(f'{path}: per-image predictions required; archived summary CSVs cannot substitute')
        frames[model] = pd.read_csv(path)
        prompt_columns=['family','variant','category','prompt_index','prompt']
        prompts=pd.read_csv(root/model/'all_prompts.csv')[prompt_columns].sort_values(prompt_columns[:-1]).reset_index(drop=True)
        split_columns=['path','label','split']
        split=pd.read_csv(root/model/'split_manifest.csv')[split_columns].sort_values('path').reset_index(drop=True)
        if reference_prompts is None:
            reference_prompts,reference_split=prompts,split
        elif not prompts.equals(reference_prompts) or not split.equals(reference_split):
            raise ValueError(f'{model}: prompts or split manifest differ across checkpoints')
    models, labels, variants, paths, truth, predictions = align_predictions(frames)
    observed, sampled = bootstrap_family_scores(truth, predictions, len(labels), args.bootstrap, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    contrast_rows(models, observed, sampled, args.dataset).to_csv(out/'paired_image_intervals.csv', index=False)
    pd.concat([f.assign(model=m) for m,f in frames.items()], ignore_index=True).to_csv(out/'source_predictions.csv', index=False)
    (out/'analysis_metadata.json').write_text(json.dumps(dict(
        dataset=args.dataset, models=models, variants=[int(v) for v in variants],
        n_images=len(paths), classes=labels, n_resamples=args.bootstrap, seed=args.seed,
        uncertainty='Images resampled jointly across all model/family/variant cells; authored sets fixed',
        multiplicity='Intervals unadjusted; no family-wide significance claim'), indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--results', default='results')
    p.add_argument('--dataset', required=True, choices=['realwaste','trashnet'])
    p.add_argument('--out', required=True)
    p.add_argument('--bootstrap', type=int, default=5000)
    p.add_argument('--seed', type=int, default=20261017)
    run(p.parse_args())
