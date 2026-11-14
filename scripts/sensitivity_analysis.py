#!/usr/bin/env python3
"""Reproduce conservative flagged-test exclusion from linguistic control predictions.

This is a *sensitivity analysis*, NOT a claim that flagged pairs are true duplicates.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score


def score(df,baseline,treatment):
    labels=sorted(df.true_label.unique())
    y=df.true_label.to_numpy();a=df[baseline].to_numpy();b=df[treatment].to_numpy()
    return dict(n=len(df),baseline_accuracy=accuracy_score(y,a),alternative_accuracy=accuracy_score(y,b),gain_pp=100*(accuracy_score(y,b)-accuracy_score(y,a)),baseline_macro_f1=f1_score(y,a,labels=labels,average='macro',zero_division=0),alternative_macro_f1=f1_score(y,b,labels=labels,average='macro',zero_division=0))


def run(args):
    df=pd.read_csv(args.predictions)
    flags=pd.read_csv(args.flags)
    flags=flags[(flags.dataset.str.lower()==args.dataset)&(flags.partition=='test')]
    def key(p):return '/'.join(str(p).replace('\\','/').split('/')[-2:])
    df['portable_path']=df.path.map(key)
    excluded=set(flags.path.map(key))
    found=set(df.portable_path)&excluded
    if len(found)!=len(excluded):raise ValueError(f'Only {len(found)} of {len(excluded)} flagged test images found in predictions')
    filtered=df[~df.portable_path.isin(excluded)]
    baseline='repeated_class_name' if 'repeated_class_name' in df else 'class_name_3'
    treatment='alternate_category_phrases' if 'alternate_category_phrases' in df else 'paraphrases_3'
    result=pd.DataFrame([dict(dataset=args.dataset,scenario='original',**score(df,baseline,treatment)),dict(dataset=args.dataset,scenario='exclude_all_flagged_test_images',**score(filtered,baseline,treatment))])
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);result.to_csv(out,index=False)
    print(result.to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--flags',default='reference/audit/flagged_pairs.csv');p.add_argument('--dataset',choices=['realwaste','trashnet'],required=True);p.add_argument('--out',required=True)
    run(p.parse_args())
