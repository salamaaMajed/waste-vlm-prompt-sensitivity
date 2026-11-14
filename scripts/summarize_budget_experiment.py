#!/usr/bin/env python3
"""Paired support-draw uncertainty; all results condition on the fixed test set."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t,ttest_1samp


def holm(p):
    p=np.asarray(p,dtype=float)
    order=np.argsort(p)
    adjusted=np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p))))
    out=np.empty(len(p));out[order]=np.minimum(adjusted,1)
    return out


def summarize(root,out):
    root=Path(root);out=Path(out)
    files=sorted(root.rglob('metrics_by_run.csv'))
    if not files:raise ValueError('No budget experiment metrics found')
    frames=[];settings=[]
    for file in files:
        meta=json.loads((file.parent/'metadata.json').read_text())
        if meta.get('status')!='complete':raise ValueError(f'Incomplete experiment: {file.parent}')
        settings.append(meta)
        frames.append(pd.read_csv(file))
    if len({m['primary_metric'] for m in settings})!=1:
        raise ValueError('Do not pool experiments with different selection objectives')
    data=pd.concat(frames,ignore_index=True)
    keys=['dataset','evaluation','backbone','budget_per_class','repeat','method']
    if data.duplicated(keys).any():raise ValueError('Duplicate run keys; summarize seed/prompt studies separately')
    if (data.additional_validation_labels!=0).any():raise ValueError('Unbudgeted validation labels recorded')
    observed=data.groupby(keys[:-2]+['method'],as_index=False).agg(
       accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),
       macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'),n_draws=('repeat','size'))
    rows=[]
    primary_metric=settings[0]['primary_metric']
    for group,frame in data.groupby(keys[:4]):
        dataset,evaluation,backbone,budget=group
        for metric in ['accuracy','macro_f1']:
            wide=frame.pivot(index='repeat',columns='method',values=metric)
            if wide.isna().any().any():raise ValueError('Unpaired method repetitions')
            for baseline in sorted(set(wide)-{'target_selected_adapter'}):
                diff=100*(wide.target_selected_adapter-wide[baseline])
                n=len(diff);mean=float(diff.mean());sd=float(diff.std(ddof=1)) if n>1 else float('nan')
                half=float(t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else float('nan')
                if n<2:pvalue=float('nan')
                elif np.allclose(diff,diff.iloc[0]):pvalue=1.0 if np.isclose(mean,0) else 0.0
                else:pvalue=float(ttest_1samp(diff,0).pvalue)
                rows.append(dict(dataset=dataset,evaluation=evaluation,backbone=backbone,budget_per_class=budget,
                    metric=metric,baseline=baseline,n_draws=n,mean_gain_pp=mean,ci95_low_pp=mean-half,ci95_high_pp=mean+half,
                    p_value=pvalue,primary=(metric==primary_metric and baseline=='pooled_ensemble_adapter'),
                    scope='Support-draw t interval; fixed test partition; pointwise unadjusted'))
    comparisons=pd.DataFrame(rows);comparisons['primary_holm_p']=np.nan
    # Correct across all supplied primary dataset/checkpoint/budget/evaluation cells.
    mask=comparisons.primary & comparisons.p_value.notna()
    comparisons.loc[mask,'primary_holm_p']=holm(comparisons.loc[mask,'p_value'])
    out.mkdir(parents=True,exist_ok=True)
    observed.to_csv(out/'method_summary.csv',index=False)
    comparisons.to_csv(out/'paired_comparisons.csv',index=False)
    (out/'SCOPE.txt').write_text(
        'Primary contrast: target-selected adapter minus pooled-ensemble adapter; metric fixed before the run.\n'
        'Primary p values Holm-adjusted across all supplied primary cells; intervals remain pointwise.\n'
        'Conditional on fixed test images, source partitions, prompts and checkpoints.\n'
        'Overlapping dataset re-splits and prompt studies must be summarized separately.\n'
        'Historical benchmark extensions are retrospective, even if code/settings were frozen before these additional runs.\n')
    print(comparisons[comparisons.primary].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',default='budget_results');p.add_argument('--out',default='budget_summary')
    args=p.parse_args();summarize(args.results,args.out)
