#!/usr/bin/env python3
"""Independently recompute inner-CV grids from exported frozen features.

Feature exports use relative image IDs; this does not need image pixels or a GPU.
It validates selection arithmetic, not the origin or correctness of the features.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from wastevlm.budget import cv_search


def audit(results,features,first_draw_only=False):
    results=Path(results);features=Path(features)
    meta=json.loads((results/'metadata.json').read_text())
    if meta['status']!='complete':raise ValueError('Incomplete experiment')
    split=pd.read_csv(results/'split_manifest.csv');pool=pd.read_csv(results/'labeled_pool.csv')
    saved=pd.read_csv(results/'cv_scores.csv');nchecked=0
    for model in meta['backbones']:
        with np.load(features/meta['dataset']/f'{model}.npz',allow_pickle=False) as z:
            x=z['features'];paths=z['paths'].tolist()
            prototypes={key:z['prototypes'][i] for i,key in enumerate(z['prompt_ids'].tolist())}
            feature_meta=json.loads(str(z['metadata'].item()))
        if feature_meta['prompt_sha256']!=meta['prompt_sha256'] or set(paths)!=set(split.path):
            raise ValueError('Wrong feature export or prompt inventory')
        index={path:i for i,path in enumerate(paths)}
        for (budget,repeat),g in pool.groupby(['budget_per_class','repeat'],sort=True):
            if first_draw_only and (budget!=meta['budgets'][0] or repeat!=0):continue
            ids=np.array([index[p] for p in g.path]);y=np.array([meta['classes'].index(c) for c in g.label])
            folds=[(np.flatnonzero(g.inner_fold.to_numpy()!=i),np.flatnonzero(g.inner_fold.to_numpy()==i)) for i in range(4)]
            _,records,_=cv_search(x[ids],y,prototypes,folds,meta['primary_metric'])
            actual=pd.DataFrame(records)
            expected=saved[(saved.backbone==model)&(saved.budget_per_class==budget)&(saved.repeat==repeat)]
            merged=actual.merge(expected,on=['prompt','beta','alpha'],suffixes=('_calc','_saved'),validate='one_to_one')
            if len(merged)!=len(actual) or len(expected)!=len(actual):raise ValueError('CV grid coverage mismatch')
            for metric in ['cv_accuracy','cv_macro_f1']:
                if not np.allclose(merged[metric+'_calc'],merged[metric+'_saved'],atol=1e-12,rtol=0):
                    raise ValueError(f'{model}/{budget}/{repeat}: recomputed CV scores differ; inspect numerical or provenance mismatch')
            nchecked+=1
    report=dict(status='passed',cv_grids_recomputed=nchecked,first_draw_only=first_draw_only,
                limitation='Frozen feature arithmetic checked; model inference and image provenance not independently verified')
    name='feature_audit_first_draw.json' if first_draw_only else 'feature_audit_full.json'
    (results/name).write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--features',required=True)
    p.add_argument('--first-draw-only',action='store_true');a=p.parse_args();audit(a.results,a.features,a.first_draw_only)
