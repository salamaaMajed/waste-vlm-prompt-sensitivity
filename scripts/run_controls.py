#!/usr/bin/env python3
"""Reproduce the Experiment 8 zero-shot conditions or 3-way linguistic follow-up."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from wastevlm.core import discover,split,relative_manifest,validate_reference_split,metrics
from wastevlm.controls import experiment8_prompts,linguistic_prompts
from wastevlm.encoders import Encoder
from wastevlm.io import load_or_encode
from wastevlm.env import environment


def paired_bootstrap(y,a,b,seed,nboot):
    rng=np.random.default_rng(seed)
    diff=(a==y).astype(float)-(b==y).astype(float)
    vals=np.array([diff[rng.integers(0,len(y),size=len(y))].mean() for _ in range(nboot)])
    return 100*float(diff.mean()),*[100*float(v) for v in np.quantile(vals,[.025,.975])]


def run(args):
    np.random.seed(args.seed);torch.manual_seed(args.seed)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    df,labels=discover(args.data,args.dataset)
    manifest,y,train,val,test=split(df,labels,args.seed)
    validate_reference_split(manifest,args.data,args.reference_split)
    relative_manifest(manifest,args.data).to_csv(out/'split_manifest.csv',index=False)
    mapping=experiment8_prompts(labels,args.seed) if args.protocol=='experiment8' else linguistic_prompts(labels)
    (out/'prompts.json').write_text(json.dumps(mapping,indent=2))
    encoder=Encoder('clip');x=load_or_encode(encoder,df.path.tolist(),args.cache_dir or out,args.batch)
    preds={};prototypes={};rows=[];class_rows=[]
    for condition,groups in mapping.items():
        proto=encoder.prototypes(groups,labels)
        prototypes[condition]=proto
        yp=(x[test]@proto.T).argmax(1)
        preds[condition]=yp
        rows.append(dict(condition=condition,n=len(test),**metrics(y[test],yp,len(labels))))
        for c,label in enumerate(labels):
            idx=(y[test]==c)
            class_rows.append(dict(condition=condition,**{'class':label},n=int(idx.sum()),recall=float((yp[idx]==c).mean())))
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(class_rows).to_csv(out/'per_class_recall.csv',index=False)
    pred_df=pd.DataFrame({'path':[Path(df.path.iloc[i]).relative_to(Path(args.data).resolve()).as_posix() for i in test], 'true_label':[labels[v] for v in y[test]]})
    for condition,yp in preds.items():pred_df[condition]=[labels[v] for v in yp]
    pred_df.to_csv(out/'predictions.csv',index=False)
    if args.protocol=='linguistic':
        comparisons=[('class_name_preserving_templates','repeated_class_name'),('alternate_category_phrases','class_name_preserving_templates')]
        rows=[]
        for a,b in comparisons:
            gain,lo,hi=paired_bootstrap(y[test],preds[a],preds[b],args.seed,args.bootstrap)
            rows.append(dict(comparison=a+' minus '+b,gain_pp=gain,ci_low_pp=lo,ci_high_pp=hi))
        pd.DataFrame(rows).to_csv(out/'paired_comparisons.csv',index=False)
        shifts=[]
        for j,c in enumerate(labels):
            shifts.append(dict(class_name=c,cosine_name_to_templates=float(prototypes['repeated_class_name'][j]@prototypes['class_name_preserving_templates'][j]),cosine_name_to_alternates=float(prototypes['repeated_class_name'][j]@prototypes['alternate_category_phrases'][j])))
        pd.DataFrame(shifts).to_csv(out/'text_prototype_similarity.csv',index=False)
    if args.protocol=='experiment8':
        rows_boot=[]
        for condition in preds:
            if condition=='attributes_3':continue
            gain,lo,hi=paired_bootstrap(y[test],preds['attributes_3'],preds[condition],args.seed,args.bootstrap)
            rows_boot.append(dict(comparison='attributes_3 minus '+condition,accuracy_gain=gain/100,ci_low=lo/100,ci_high=hi/100))
        pd.DataFrame(rows_boot).to_csv(out/'paired_bootstrap.csv',index=False)
    (out/'metadata.json').write_text(json.dumps(dict(dataset=args.dataset,protocol=args.protocol,seed=args.seed,n_images=len(df),n_test=len(test),bootstrap=args.bootstrap,checkpoint='openai/clip-vit-base-patch16',environment=environment()),indent=2))
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--dataset',choices=['realwaste','trashnet'],required=True)
    p.add_argument('--protocol',choices=['experiment8','linguistic'],required=True)
    p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--batch',type=int,default=16);p.add_argument('--cache-dir',default='')
    p.add_argument('--bootstrap',type=int,default=5000);p.add_argument('--reference-split',default='')
    run(p.parse_args())
