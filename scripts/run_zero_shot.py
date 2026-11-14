#!/usr/bin/env python3
"""Reproduce Experiments 10-12: 9 prompt sets x 3 model families x 2 datasets."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from wastevlm.core import discover, split, relative_manifest, validate_reference_split, metrics
from wastevlm.encoders import Encoder, CHECKPOINTS
from wastevlm.io import load_or_encode
from wastevlm.env import environment
from wastevlm.prompts import prompt_sets


def run(args):
    np.random.seed(args.seed);torch.manual_seed(args.seed)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    df,labels=discover(args.data,args.dataset)
    manifest,y,train,val,test=split(df,labels,args.seed)
    validate_reference_split(manifest,args.data,args.reference_split)
    relative_manifest(manifest,args.data).to_csv(out/'split_manifest.csv',index=False)
    encoder=Encoder(args.backbone)
    x=load_or_encode(encoder,df.path.tolist(),args.cache_dir or out,args.batch)
    groups=prompt_sets(labels)
    if args.prompt_csv:
        from wastevlm.custom_prompts import read_prompt_csv
        groups=read_prompt_csv(args.prompt_csv,labels)
    rows=[];pred=[];prompt_rows=[];validation_rows=[]
    for (family,variant),mapping in groups.items():
        for c in labels:
            for j,prompt in enumerate(mapping[c]):
                prompt_rows.append(dict(family=family,variant=variant,category=c,prompt_index=j,prompt=prompt))
        p=encoder.prototypes(mapping,labels)
        predicted=(x[test]@p.T).argmax(axis=1)
        score=metrics(y[test],predicted,len(labels))
        validation=metrics(y[val],(x[val]@p.T).argmax(axis=1),len(labels))
        validation_rows.append(dict(family=family,variant=variant,**validation))
        rows.append(dict(dataset=args.dataset,model=args.backbone,family=family,variant=variant,n_test=len(test),**score))
        for i,guess in zip(test,predicted):
            pred.append(dict(dataset=args.dataset,path=Path(df.path.iloc[i]).relative_to(Path(args.data).resolve()).as_posix(),family=family,variant=variant,true_label=labels[y[i]],predicted_label=labels[guess],correct=int(y[i]==guess)))
        print(f'{args.dataset} {args.backbone} {family} set {variant}: accuracy={score["accuracy"]:.4f}',flush=True)
    frame=pd.DataFrame(rows)
    validation_frame=pd.DataFrame(validation_rows)
    validation_frame.to_csv(out/'validation_metrics.csv',index=False)
    selection=[]
    for metric in ('accuracy','macro_f1'):
        # Stable ordering breaks validation ties; test scores never select a prompt.
        winner=validation_frame.sort_values([metric,'family','variant'],ascending=[False,True,True]).iloc[0]
        test_row=frame[(frame.family==winner.family)&(frame.variant==winner.variant)].iloc[0]
        selection.append(dict(selection_metric=metric,family=winner.family,variant=int(winner.variant),
                              validation_score=winner[metric],test_accuracy=test_row.accuracy,
                              test_macro_f1=test_row.macro_f1,
                              scope='Target-label validation used for prompt selection; not label-free zero-shot selection'))
    pd.DataFrame(selection).to_csv(out/'validation_selected_results.csv',index=False)
    frame.to_csv(out/'metrics_by_prompt_set.csv',index=False)
    pd.DataFrame(pred).to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(prompt_rows).to_csv(out/'all_prompts.csv',index=False)
    summary=frame.groupby('family',as_index=False).agg(accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'),sets=('variant','count'))
    summary.to_csv(out/'family_summary.csv',index=False)
    (out/'metadata.json').write_text(json.dumps(dict(dataset=args.dataset,backbone=args.backbone,checkpoint=CHECKPOINTS[args.backbone],seed=args.seed,n_images=len(df),n_train=len(train),n_validation=len(val),n_test=len(test),classes=labels,reference_split_verified=bool(args.reference_split),environment=environment(),prompt_csv=args.prompt_csv,notes='All fixed configurations reported. Additional validation-selected results are distinct from fixed-prompt zero-shot results.'),indent=2))
    print(summary.to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--dataset',choices=['realwaste','trashnet'],required=True)
    p.add_argument('--backbone',choices=list(CHECKPOINTS),required=True)
    p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--batch',type=int,default=16);p.add_argument('--cache-dir',default='');p.add_argument('--reference-split',default='')
    p.add_argument('--prompt-csv',default='',help='Frozen user-supplied prompt CSV for a NEW experiment; omit to reproduce historical prompts')
    run(p.parse_args())
