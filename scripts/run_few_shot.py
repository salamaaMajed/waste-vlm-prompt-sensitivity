#!/usr/bin/env python3
"""Reproduce Experiment 9: matched 30-draw CLIP Tip-Adapter comparison.

The text prototypes here intentionally follow the *original Experiment 9*:
  baseline: a photo of <class> waste
  AE-Tip: a photo of <each of three object-oriented DESCRIPTIONS>
This differs from Experiment 10's three-family prompt grid.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t
from wastevlm.core import discover,split,relative_manifest,validate_reference_split,metrics,cache_logits
from wastevlm.encoders import Encoder, CHECKPOINTS
from wastevlm.io import load_or_encode
from wastevlm.env import environment
from wastevlm.prompts import DESCRIPTIONS


def run(args):
    np.random.seed(args.seed);torch.manual_seed(args.seed)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    df,labels=discover(args.data,args.dataset)
    manifest,y,train,val,test=split(df,labels,args.seed)
    validate_reference_split(manifest,args.data,args.reference_split)
    relative_manifest(manifest,args.data).to_csv(out/'split_manifest.csv',index=False)
    for c in range(len(labels)):
        if (y[train]==c).sum()<max(args.shots):raise ValueError('Not enough training images for requested shots')
    encoder=Encoder(args.backbone)
    x=load_or_encode(encoder,df.path.tolist(),args.cache_dir or out,args.batch)
    basic={c:[f'a photo of {c} waste'] for c in labels}
    semantic={c:[f'a photo of {d}' for d in DESCRIPTIONS[c]] for c in labels}
    basic_logits=100*x@encoder.prototypes(basic,labels).T
    ae_logits=100*x@encoder.prototypes(semantic,labels).T
    rows=[];predictions=[];selected=[];support_records=[]
    baseline_method='tip_adapter_matched' if args.tuning_mode=='baseline' else 'tip_adapter_independently_tuned'
    alternative_method=('alternative_expression_tip_adapter_matched' if args.tuning_mode=='baseline'
                        else 'alternative_expression_tip_adapter_independently_tuned')
    root=Path(args.data).resolve()
    def evaluate(method,shots,repeat,ypred):
        score=metrics(y[test],ypred,len(labels))
        rows.append(dict(dataset=args.dataset,method=method,shots=shots,repeat=repeat,n_test=len(test),**score))
        for j,guess in zip(test,ypred):
            predictions.append(dict(dataset=args.dataset,path=Path(df.path.iloc[j]).relative_to(root).as_posix(),true_label=labels[y[j]],prediction=labels[guess],method=method,shots=shots,repeat=repeat,correct=int(y[j]==guess)))
    evaluate('zero_shot_class_name',0,0,basic_logits[test].argmax(1))
    evaluate('zero_shot_semantic',0,0,ae_logits[test].argmax(1))
    for shots in args.shots:
        for rep in range(args.repeats):
            support_seed=args.seed+1009*rep+97*shots
            rng=np.random.default_rng(support_seed)
            chosen=np.concatenate([rng.choice(train[y[train]==c],size=shots,replace=False) for c in range(len(labels))])
            for j in chosen:
                support_records.append(dict(dataset=args.dataset,shots=shots,repeat=rep,seed=support_seed,path=Path(df.path.iloc[j]).relative_to(root).as_posix(),label=df.label.iloc[j]))
            # Historical search order and strict '>' tie-breaking retained.
            best=(-1,None,None)
            best_alt=(-1,None,None)
            for beta in [1,5,10,20]:
                vcache=cache_logits(x[val],x[chosen],y[chosen],beta,len(labels))
                for alpha in [1,5,10,20,50]:
                    acc=metrics(y[val],(basic_logits[val]+alpha*vcache).argmax(1),len(labels))['accuracy']
                    if acc>best[0]:best=(acc,beta,alpha)
                    if args.tuning_mode == 'independent':
                        alt_acc=metrics(y[val],(ae_logits[val]+alpha*vcache).argmax(1),len(labels))['accuracy']
                        if alt_acc>best_alt[0]:best_alt=(alt_acc,beta,alpha)
            val_acc,beta,alpha=best
            alt_val_acc,alt_beta,alt_alpha=best_alt if args.tuning_mode=='independent' else (None,beta,alpha)
            selected.append(dict(shots=shots,repeat=rep,beta=beta,alpha=alpha,baseline_validation_accuracy=val_acc,
                                 alternative_beta=alt_beta,alternative_alpha=alt_alpha,
                                 alternative_validation_accuracy=alt_val_acc,tuning_mode=args.tuning_mode))
            tcache=cache_logits(x[test],x[chosen],y[chosen],beta,len(labels))
            evaluate(baseline_method,shots,rep,(basic_logits[test]+alpha*tcache).argmax(1))
            alt_cache=tcache if alt_beta==beta else cache_logits(x[test],x[chosen],y[chosen],alt_beta,len(labels))
            evaluate(alternative_method,shots,rep,(ae_logits[test]+alt_alpha*alt_cache).argmax(1))
            print(f'{args.dataset} {shots}-shot draw {rep+1}/{args.repeats}',flush=True)
    run=pd.DataFrame(rows);run.to_csv(out/'metrics_by_run.csv',index=False)
    summary=run.groupby(['method','shots'],as_index=False).agg(accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'),repeats=('accuracy','size'))
    summary.to_csv(out/'summary.csv',index=False)
    pd.DataFrame(predictions).to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(selected).to_csv(out/'matched_hyperparameters.csv',index=False)
    pd.DataFrame(support_records).to_csv(out/'support_set_ids.csv',index=False)
    paired=[]
    for shots in args.shots:
        a=run[(run.method==baseline_method)&(run.shots==shots)].sort_values('repeat')
        b=run[(run.method==alternative_method)&(run.shots==shots)].sort_values('repeat')
        assert np.array_equal(a.repeat.to_numpy(),b.repeat.to_numpy())
        for metric in ['accuracy','macro_f1']:
            diff=(b[metric].to_numpy()-a[metric].to_numpy())*100
            mean=float(diff.mean());sd=float(diff.std(ddof=1)) if len(diff)>1 else float('nan')
            half=float(student_t.ppf(.975,len(diff)-1)*sd/np.sqrt(len(diff))) if len(diff)>1 else float('nan')
            paired.append(dict(dataset=args.dataset,shots=shots,metric=metric,n_draws=len(diff),mean_gain_pp=mean,sd_gain_pp=sd,ci95_low_pp=mean-half,ci95_high_pp=mean+half,interval_scope='Support-set draws; fixed test images; unadjusted'))
    pd.DataFrame(paired).to_csv(out/'paired_draw_intervals.csv',index=False)
    protocol=('Historical matched intervention; baseline-selected alpha,beta transferred unchanged' if args.tuning_mode=='baseline'
              else 'NEW EXPERIMENT: each text treatment independently validation-tuned; identical support images')
    (out/'metadata.json').write_text(json.dumps(dict(dataset=args.dataset,model=CHECKPOINTS[args.backbone],
        backbone=args.backbone,tuning_mode=args.tuning_mode,text_logit_scale=100,seed=args.seed,n_images=len(df),
        n_train=len(train),n_validation=len(val),n_test=len(test),classes=labels,shots=args.shots,repeats=args.repeats,
        support_ids_logged=True,environment=environment(),protocol=protocol,
        method_name_note='Independent tuning uses distinct method IDs and must not be compared to historical reference tables as a reproduction',
        published_protocol=(args.backbone=='clip' and args.tuning_mode=='baseline' and args.seed==42)),indent=2))
    print(summary.to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--dataset',choices=['realwaste','trashnet'],required=True)
    p.add_argument('--out',required=True);p.add_argument('--shots',type=int,nargs='+',default=[1,2,4,8,16]);p.add_argument('--repeats',type=int,default=30)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--batch',type=int,default=16);p.add_argument('--cache-dir',default='');p.add_argument('--reference-split',default='')
    p.add_argument('--backbone', choices=list(CHECKPOINTS), default='clip')
    p.add_argument('--tuning-mode', choices=['baseline','independent'], default='baseline',
                   help='Independent tuning is a new experiment, not a reproduction of the paper tables')
    run(p.parse_args())
