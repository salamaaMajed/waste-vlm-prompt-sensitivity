#!/usr/bin/env python3
"""Recompute every saved internal test prediction from portable frozen features.

This audits feature arithmetic and selected configurations, not fresh image
encoding. It deliberately does not use test labels to choose configurations.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from wastevlm.budget import Choice, predict_choice, centroid_predict


def audit(results, features):
    results, features = Path(results), Path(features)
    meta = json.loads((results / 'metadata.json').read_text())
    split = pd.read_csv(results / 'split_manifest.csv')
    pools = pd.read_csv(results / 'labeled_pool.csv')
    choices = pd.read_csv(results / 'selected_configs.csv')
    checked = decisions = 0
    # Read only a single checkpoint's predictions at a time.
    for model in meta['backbones']:
        chunks = []
        for chunk in pd.read_csv(results / 'predictions.csv', chunksize=200000):
            chunks.append(chunk[(chunk.backbone == model) & (chunk.evaluation == 'internal')])
        saved = pd.concat(chunks, ignore_index=True)
        saved_groups = {key: frame.set_index('path') for key, frame in
                        saved.groupby(['budget_per_class', 'repeat', 'method'])}
        with np.load(features / meta['dataset'] / f'{model}.npz', allow_pickle=False) as z:
            x = z['features']
            index = {p: i for i, p in enumerate(z['paths'].tolist())}
            prototypes = dict(zip(z['prompt_ids'].tolist(), z['prototypes']))
        tests = split[split.split == 'test'].path.tolist()
        query = x[[index[p] for p in tests]]
        for (budget, repeat), pool in pools.groupby(['budget_per_class', 'repeat']):
            support = x[[index[p] for p in pool.path]]
            y = np.array([meta['classes'].index(c) for c in pool.label])
            sim = np.clip(query @ support.T, -1, 1)
            configs = choices[(choices.backbone == model) &
                              (choices.budget_per_class == budget) & (choices.repeat == repeat)]
            # A centroid has no selected hyperparameters, hence no row in
            # selected_configs.csv. Audit its predictions explicitly too.
            configs = pd.concat([configs, pd.DataFrame([dict(method='nearest_centroid')])],
                                ignore_index=True)
            for _, row in configs.iterrows():
                if row.method == 'nearest_centroid':
                    pred = centroid_predict(query, support, y, len(meta['classes']))
                else:
                    choice = Choice(row.prompt, row.beta, row.alpha, row.cv_accuracy, row.cv_macro_f1)
                    pred = predict_choice(query, support, y, prototypes, choice, similarities=sim)
                actual = saved_groups[(budget, repeat, row.method)].loc[tests]
                expected = np.array(meta['classes'])[pred]
                if not np.array_equal(actual.prediction.to_numpy(), expected):
                    count = int(np.count_nonzero(actual.prediction.to_numpy() != expected))
                    raise ValueError(f'{model}/{budget}/{repeat}/{row.method}: {count} predictions differ')
                checked += 1
                decisions += len(tests)
    expected_runs = len(meta['backbones']) * len(meta['budgets']) * meta['repeats'] * 8
    if checked != expected_runs:
        raise ValueError(f'Only {checked} of {expected_runs} method runs checked')
    report = dict(status='passed', dataset=meta['dataset'], method_runs_recomputed=checked,
                  test_decisions_recomputed=decisions,
                  limitation='Frozen feature arithmetic; no independent image-encoding rerun')
    (results / 'test_feature_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    parser.add_argument('--features', required=True)
    args = parser.parse_args()
    audit(args.results, args.features)
