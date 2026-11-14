#!/usr/bin/env python3
"""Compare fresh result CSVs against archived paper summaries; never silently replace values."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def compare(actual, expected, keys, cols, tolerance):
    a=pd.read_csv(actual);e=pd.read_csv(expected)
    merged=e.merge(a,on=keys,how='outer',suffixes=('_reference','_new'),indicator=True,validate='one_to_one')
    if not (merged['_merge']=='both').all():
        raise AssertionError(f'Missing/unexpected rows: {merged[merged._merge!="both"][keys].to_dict("records")}')
    for col in cols:
        ref=merged[col+'_reference'].to_numpy(float)
        new=merged[col+'_new'].to_numpy(float)
        max_diff=float(np.max(np.abs(ref-new)))
        print(f'{actual.parent.name} {col}: max absolute difference {max_diff:.7f}')
        if not np.allclose(ref,new,rtol=0,atol=tolerance):
            raise AssertionError(f'{col} differs beyond tolerance {tolerance}')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',required=True);p.add_argument('--reference',default='reference')
    p.add_argument('--tolerance',type=float,default=0.002,help='Absolute tolerance on 0-1 accuracy/F1 and pp intervals. Tighten if environment matches.')
    args=p.parse_args();root=Path(args.results);ref=Path(args.reference)
    checked=0
    for dataset in ('realwaste','trashnet'):
        for model in ('clip','siglip','openclip'):
            actual=root/'zero_shot'/dataset/model/'family_summary.csv'
            expected=ref/'zero_shot'/dataset/model/'family_summary.csv'
            if actual.exists():
                compare(actual,expected,['family'],['accuracy_mean','macro_f1_mean'],args.tolerance)
                checked+=1
        actual=root/'few_shot'/dataset/'paired_draw_intervals.csv'
        expected=ref/'few_shot'/dataset/'paired_draw_intervals.csv'
        if actual.exists():
            compare(actual,expected,['shots','metric'],['mean_gain_pp','ci95_low_pp','ci95_high_pp'],args.tolerance*100)
            checked+=1
        for protocol in ('experiment8','linguistic'):
            actual=root/'controls'/dataset/protocol/'metrics.csv'
            expected=ref/'controls'/dataset/protocol/'metrics.csv'
            if actual.exists():
                compare(actual,expected,['condition'],['accuracy','macro_f1'],args.tolerance)
                checked+=1
    if checked==0:raise SystemExit('No result CSVs found under expected directory structure')
    print(f'PASS: checked {checked} result groups against archived paper summaries')

if __name__=='__main__':main()
