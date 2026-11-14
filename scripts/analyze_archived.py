#!/usr/bin/env python3
"""Derive set-omission diagnostics from measured reference summaries.

No model inference or image-level bootstrap is performed by this script.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def run(reference, out):
    reference, out = Path(reference), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows=[]
    for dataset in ('realwaste','trashnet'):
        for model in ('clip','siglip','openclip'):
            frame=pd.read_csv(reference/'zero_shot'/dataset/model/'metrics_by_prompt_set.csv')
            for metric in ('accuracy','macro_f1'):
                grid=frame.pivot(index='variant', columns='family', values=metric)
                delta=100*(grid.object_expressions-grid.name_preserving)
                if len(delta)!=3 or delta.isna().any():
                    raise ValueError('Expected complete three-variant archive')
                loo=np.array([(delta.sum()-v)/(len(delta)-1) for v in delta])
                rows.append(dict(dataset=dataset,model=model,metric=metric,
                                 variant_1_pp=delta.loc[1],variant_2_pp=delta.loc[2],variant_3_pp=delta.loc[3],
                                 mean_pp=delta.mean(),set_sd_pp=delta.std(ddof=1),
                                 loo_min_pp=loo.min(),loo_max_pp=loo.max(),
                                 all_variant_signs_agree=bool((delta>0).all() or (delta<0).all()),
                                 loo_changes_mean_sign=bool((loo*delta.mean()<0).any())))
    pd.DataFrame(rows).to_csv(out/'authored_set_sensitivity.csv',index=False)
    tables=[]
    for dataset in ('realwaste','trashnet'):
        frame=pd.read_csv(reference/'few_shot'/dataset/'summary.csv')
        table=[]
        for shots in (0,1,2,4,8,16):
            names=('zero_shot_class_name','zero_shot_semantic') if shots==0 else (
                'tip_adapter_matched','alternative_expression_tip_adapter_matched')
            a=frame[(frame.shots==shots)&(frame.method==names[0])].iloc[0]
            b=frame[(frame.shots==shots)&(frame.method==names[1])].iloc[0]
            table.append(dict(dataset=dataset,shots=shots,name_accuracy=100*a.accuracy_mean,
                              alternative_accuracy=100*b.accuracy_mean,accuracy_gain_pp=100*(b.accuracy_mean-a.accuracy_mean),
                              name_macro_f1=100*a.macro_f1_mean,alternative_macro_f1=100*b.macro_f1_mean,
                              macro_f1_gain_pp=100*(b.macro_f1_mean-a.macro_f1_mean)))
        tables.extend(table)
    pd.DataFrame(tables).to_csv(out/'cache_trajectory_exact_prototypes.csv',index=False)
    (out/'STATUS.txt').write_text(
        'Derived from archived measured per-set and few-shot summaries. No fresh inference.\n'
        'Leave-one-set-out ranges are descriptive, not confidence intervals.\n'
        'Run analyze_cross_backbone.py on recovered or freshly generated predictions for paired image intervals.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--reference',default='reference')
    p.add_argument('--out',default='revision_analysis')
    args=p.parse_args()
    run(args.reference,args.out)
