#!/usr/bin/env python3
"""Audit saved label budgets, CV choices, and reported test metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score


def audit(folder):
    folder=Path(folder);meta=json.loads((folder/'metadata.json').read_text())
    if meta['status']!='complete':raise ValueError('Incomplete experiment')
    pool=pd.read_csv(folder/'labeled_pool.csv');split=pd.read_csv(folder/'split_manifest.csv')
    config=pd.read_csv(folder/'selected_configs.csv');cv=pd.read_csv(folder/'cv_scores.csv')
    metrics=pd.read_csv(folder/'metrics_by_run.csv');oof=pd.read_csv(folder/'selected_oof_predictions.csv')
    predictions=pd.read_csv(folder/'predictions.csv');source=pd.read_csv(folder/'source_clip_choices.csv')
    classes=meta['classes'];partition=dict(zip(split.path,split.split));truth=dict(zip(split.path,split.label))
    if pool.duplicated(['budget_per_class','repeat','path']).any():raise ValueError('Duplicate pool image within a draw')
    if not all(partition.get(p)=='train' for p in pool.path):raise ValueError('A budgeted label came from held-out data')
    if not all(truth[p]==label for p,label in zip(pool.path,pool.label)):raise ValueError('Pool truth mismatch')
    expected={(b,r) for b in meta['budgets'] for r in range(meta['repeats'])}
    if set(pool.groupby(['budget_per_class','repeat']).groups)!=expected:raise ValueError('Missing pool draws')
    for (budget,repeat),g in pool.groupby(['budget_per_class','repeat']):
        if set(g.label)!=set(classes) or not (g.groupby('label').size()==budget).all():raise ValueError('Incorrect label budget')
        if set(g.inner_fold)!={0,1,2,3}:raise ValueError('Missing inner fold')
        if meta.get('group_aware'):
            groups=split.set_index('path').loc[g.path,'group_id'].to_numpy()
            folds=g.inner_fold.to_numpy()
            if (pd.DataFrame({'group':groups,'fold':folds}).groupby('group').fold.nunique()>1).any():raise ValueError('Group leakage in inner folds')
    if (config.additional_validation_labels!=0).any() or (metrics.additional_validation_labels!=0).any():raise ValueError('Extra validation labels used')
    if not np.array_equal(config.n_unique_labels,config.budget_per_class*len(classes)):raise ValueError('Incorrect per-method label accounting')
    methods={'class_name_adapter','name_ensemble_adapter','object_ensemble_adapter','pooled_ensemble_adapter',
             'clip_prompt_transferred_adapter','target_selected_adapter','target_selected_text','nearest_centroid'}
    evaluations=['internal','external'] if meta['external_test'] else ['internal']
    expected_runs={(meta['dataset'],e,m,b,r,k) for e in evaluations for m in meta['backbones']
                   for b,r in expected for k in methods}
    keys=['dataset','evaluation','backbone','budget_per_class','repeat','method']
    if metrics.duplicated(keys).any() or set(map(tuple,metrics[keys].itertuples(index=False,name=None)))!=expected_runs:
        raise ValueError('Missing or duplicate method runs')
    config_keys=['dataset','backbone','budget_per_class','repeat','method']
    expected_configs={(meta['dataset'],m,b,r,k) for m in meta['backbones'] for b,r in expected for k in methods-{'nearest_centroid'}}
    if config.duplicated(config_keys).any() or set(map(tuple,config[config_keys].itertuples(index=False,name=None)))!=expected_configs:
        raise ValueError('Missing or duplicate selected configurations')
    prompt_ids=set(pd.read_csv(folder/'frozen_prompts.csv').prompt_id)
    grid={(p,1,0) for p in prompt_ids}|{(p,b,a) for p in prompt_ids for b in [1,5,10,20] for a in [1,5,10,20,50]}
    if cv.duplicated(['backbone','budget_per_class','repeat','prompt','beta','alpha']).any():raise ValueError('Duplicate CV score rows')
    for (model,budget,repeat),g in cv.groupby(['backbone','budget_per_class','repeat']):
        if set(map(tuple,g[['prompt','beta','alpha']].itertuples(index=False,name=None)))!=grid or not (g.n_cv_labels==budget*len(classes)).all():
            raise ValueError('Incomplete CV grid or wrong CV label count')
    objective='cv_'+meta['primary_metric']
    for row in config.itertuples():
        scores=cv[(cv.backbone==row.backbone)&(cv.budget_per_class==row.budget_per_class)&(cv.repeat==row.repeat)]
        if row.method=='target_selected_adapter':scores=scores[~scores.prompt.isin(['class_name','pooled_ensemble'])]
        elif row.method=='target_selected_text':scores=scores[(scores.alpha==0)&~scores.prompt.isin(['class_name','pooled_ensemble'])]
        else:
            fixed={'class_name_adapter':'class_name','name_ensemble_adapter':'name_preserving__1',
                   'object_ensemble_adapter':'object_expressions__1','pooled_ensemble_adapter':'pooled_ensemble'}
            prompt=(source[(source.budget_per_class==row.budget_per_class)&(source.repeat==row.repeat)].iloc[0].prompt
                    if row.method=='clip_prompt_transferred_adapter' else fixed[row.method])
            if row.prompt!=prompt:raise ValueError('Wrong fixed/transferred prompt')
            scores=scores[scores.prompt==prompt]
        if not np.isclose(getattr(row,objective),scores[objective].max(),atol=1e-12):raise ValueError('Selected configuration did not maximize inner-CV metric')
        matching=scores[(scores.prompt==row.prompt)&(scores.beta==row.beta)&(scores.alpha==row.alpha)]
        if len(matching)!=1:raise ValueError('Selected configuration missing from CV grid')
        gp=oof[(oof.backbone==row.backbone)&(oof.budget_per_class==row.budget_per_class)&(oof.repeat==row.repeat)&(oof.method==row.method)]
        sampled=pool[(pool.budget_per_class==row.budget_per_class)&(pool.repeat==row.repeat)]
        if gp.path.duplicated().any() or set(gp.path)!=set(sampled.path):raise ValueError('OOF predictions do not cover exactly the labeled pool')
        bypath=sampled.set_index('path')
        if not np.array_equal(gp.true_label.to_numpy(),bypath.loc[gp.path,'label'].to_numpy()):raise ValueError('OOF truth mismatch')
        if not np.array_equal(gp.inner_fold.to_numpy(),bypath.loc[gp.path,'inner_fold'].to_numpy()):raise ValueError('OOF fold mismatch')
        acc=accuracy_score(gp.true_label,gp.prediction)
        f1=f1_score(gp.true_label,gp.prediction,labels=classes,average='macro',zero_division=0)
        if not np.allclose([acc,f1],[row.cv_accuracy,row.cv_macro_f1],atol=1e-12):raise ValueError('Recorded CV score does not match selected OOF predictions')
    lookup=metrics.set_index(keys)
    if predictions.duplicated(keys+['path']).any():raise ValueError('Duplicate test prediction keys')
    if set(predictions.groupby(keys).groups)!=expected_runs:raise ValueError('Missing prediction runs')
    external_truth=None
    if meta['external_test']:
        external=pd.read_csv(folder/'external_test_manifest.csv');external_truth=dict(zip(external.path,external.label))
    for key,g in predictions.groupby(keys):
        expected_truth=truth if key[1]=='internal' else external_truth
        expected_paths=set(split[split.split=='test'].path) if key[1]=='internal' else set(external_truth)
        if set(g.path)!=expected_paths:raise ValueError('Incorrect held-out image inventory')
        if not all(expected_truth[p]==t for p,t in zip(g.path,g.true_label)):raise ValueError('Incorrect test labels')
        if not (g.correct==(g.true_label==g.prediction).astype(int)).all():raise ValueError('Incorrect correctness flags')
        acc=accuracy_score(g.true_label,g.prediction)
        f1=f1_score(g.true_label,g.prediction,labels=classes,average='macro',zero_division=0)
        row=lookup.loc[key]
        if not np.allclose([acc,f1],[row.accuracy,row.macro_f1],atol=1e-12):raise ValueError('Saved test metrics mismatch source predictions')
    report=dict(status='passed',dataset=meta['dataset'],n_method_runs=len(metrics),
        budget_verified=True,additional_validation_labels=0,selected_cv_scores_reconstructed=True,test_metrics_reconstructed=True,
        limitation='Without feature bundles this checks accounting and output consistency, not a fresh model-inference rerun')
    (folder/'audit_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);a=p.parse_args();audit(a.results)
