#!/usr/bin/env python3
"""Fixed-label-budget selection. Historical validation labels are never used."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from wastevlm.core import discover,split,relative_manifest,validate_reference_split,metrics
from wastevlm.budget import budget_ids,inner_folds,cv_search,best_prompt,predict_choice,centroid_predict
from wastevlm.budget_features import prompt_inventory,image_fingerprint,feature_bundle


def external_images(manifest, labels):
    frame=pd.read_csv(manifest)
    if not {'path','label','source_id'}.issubset(frame):
        raise ValueError('External manifest requires path, label, source_id')
    if frame[['path','label','source_id']].isna().any().any() or frame.empty:
        raise ValueError('External manifest has missing provenance or labels')
    frame['label']=frame.label.str.lower().str.replace('_',' ').str.strip()
    if set(frame.label)!=set(labels):
        raise ValueError('External test must contain the complete, documented canonical class inventory')
    parent=Path(manifest).resolve().parent
    frame['path']=frame.path.map(lambda p:str((parent/str(p)).resolve()))
    if frame.path.duplicated().any() or not all(Path(p).is_file() for p in frame.path):
        raise ValueError('Missing or duplicate external image paths')
    return frame.sort_values('path').reset_index(drop=True)


def run(args):
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory is not empty. Use a new name to preserve completed measurements.')
    df,labels=discover(args.data,args.dataset)
    manifest,y,train,val,test=split(df,labels,args.seed)
    validate_reference_split(manifest,args.data,args.reference_split)
    if args.split_manifest:
        custom=pd.read_csv(args.split_manifest)
        portable=relative_manifest(manifest,args.data)
        if custom.path.duplicated().any() or set(custom.path)!=set(portable.path):
            raise ValueError('Custom split must cover every discovered relative image path exactly once')
        bypath=custom.set_index('path')
        if not np.array_equal(bypath.loc[portable.path,'label'].to_numpy(),df.label.to_numpy()):
            raise ValueError('Custom split labels differ from image folders')
        assignments=bypath.loc[portable.path,'split'].to_numpy()
        if set(assignments)!={'train','validation','test'}:
            raise ValueError('Expected train, validation and test split names')
        manifest['split']=assignments
        train=np.flatnonzero(assignments=='train');val=np.flatnonzero(assignments=='validation');test=np.flatnonzero(assignments=='test')
        if 'group_id' in custom:
            if custom.group_id.isna().any() or (custom.groupby('group_id').split.nunique()>1).any():
                raise ValueError('A source/duplicate group crosses partitions or has a missing ID')
    for partition in (train,val,test):
        if set(y[partition])!=set(range(len(labels))):
            raise ValueError('Every split must contain the complete class inventory')
    mappings,candidates,prompt_sha=prompt_inventory(labels,args.prompt_csv)
    fingerprint=image_fingerprint(df.path.tolist())
    external=external_images(args.external_manifest,labels) if args.external_manifest else None
    if external is not None:
        if set(external.path)&set(df.path):raise ValueError('External images overlap source paths')
        source_hashes={hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in df.path}
        external_hashes=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in external.path]
        if set(external_hashes)&source_hashes:raise ValueError('Exact file duplicates between source and external test')
        if len(set(external_hashes))!=len(external_hashes):raise ValueError('Duplicate files inside the external test')
        external_fingerprint=image_fingerprint(external.path.tolist())
    bundles={};external_bundles={}
    for model in sorted(set(args.backbones)|{'clip'}):
        print(f'Preparing {args.dataset}/{model}',flush=True)
        bundles[model]=feature_bundle(df.path.tolist(),labels,mappings,prompt_sha,model,
                     Path(args.features)/args.dataset/'source',fingerprint,args.batch)
        if external is not None:
            external_bundles[model]=feature_bundle(external.path.tolist(),labels,mappings,prompt_sha,model,
                     Path(args.features)/args.dataset/'external',external_fingerprint,args.batch)
            if any(not np.allclose(bundles[model][1][k],external_bundles[model][1][k],atol=1e-6,rtol=1e-5) for k in mappings):
                raise ValueError('Source/external text features differ; regenerate both with identical checkpoint weights')
    out.mkdir(parents=True,exist_ok=True)
    portable=relative_manifest(manifest,args.data)
    if args.split_manifest and 'group_id' in custom:portable['group_id']=bypath.loc[portable.path,'group_id'].to_numpy()
    portable.to_csv(out/'split_manifest.csv',index=False)
    if external is not None:
        external_ids=[str(Path(p).relative_to(Path(args.external_manifest).resolve().parent))
                      if Path(p).is_relative_to(Path(args.external_manifest).resolve().parent)
                      else f'external_{i:06d}/{Path(p).name}' for i,p in enumerate(external.path)]
        external_export=external.copy();external_export['path']=external_ids
        external_export.to_csv(out/'external_test_manifest.csv',index=False)
    prompt_rows=[]
    for key,mapping in mappings.items():
        for label,prompts in mapping.items():
            for i,prompt in enumerate(prompts):prompt_rows.append(dict(prompt_id=key,category=label,prompt_index=i,prompt=prompt))
    pd.DataFrame(prompt_rows).to_csv(out/'frozen_prompts.csv',index=False)
    writers={};handles=[]
    def emit(filename,row):
        if filename not in writers:
            f=(out/filename).open('w',newline='');handles.append(f)
            writers[filename]=csv.DictWriter(f,fieldnames=list(row));writers[filename].writeheader()
        writers[filename].writerow(row)
    source_choices={}
    try:
        for budget in args.budgets:
            for repeat in range(args.repeats):
                full_groups=portable.group_id.to_numpy() if 'group_id' in portable else None
                ids=budget_ids(train,y,budget,repeat,args.seed,full_groups)
                fold_seed=args.seed+1009*repeat+97*budget+313
                group_ids=(portable.group_id.to_numpy()[ids] if 'group_id' in portable else None)
                folds=inner_folds(y[ids],fold_seed,group_ids)
                fold_ids=np.empty(len(ids),dtype=int)
                for i,(_,valid) in enumerate(folds):fold_ids[valid]=i
                for local,j in enumerate(ids):
                    emit('labeled_pool.csv',dict(dataset=args.dataset,budget_per_class=budget,repeat=repeat,
                         path=portable.path.iloc[j],label=labels[y[j]],inner_fold=int(fold_ids[local])))
                # These are the same physical labels for every checkpoint; CLIP adds compute, not labels.
                source_x,source_proto,_=bundles['clip']
                source_fit,source_records,source_oof=cv_search(source_x[ids],y[ids],source_proto,folds,args.objective)
                source_choice=best_prompt(source_fit,candidates,args.objective)
                source_choices[(budget,repeat)]=source_choice
                emit('source_clip_choices.csv',dict(dataset=args.dataset,budget_per_class=budget,repeat=repeat,
                      **asdict(source_choice),selection_metric=args.objective,n_unique_labels=len(ids)))
                for model in args.backbones:
                    x,prototypes,metadata=bundles[model]
                    choices,cv_records,oof=((source_fit,source_records,source_oof) if model=='clip'
                                           else cv_search(x[ids],y[ids],prototypes,folds,args.objective))
                    for record in cv_records:
                        emit('cv_scores.csv',dict(dataset=args.dataset,backbone=model,budget_per_class=budget,repeat=repeat,**record))
                    selected=best_prompt(choices,candidates,args.objective)
                    text_only=best_prompt(choices,candidates,args.objective,text_only=True,records=cv_records)
                    methods={
                       'class_name_adapter':choices['class_name'],
                       'name_ensemble_adapter':choices['name_preserving__1'],
                       'object_ensemble_adapter':choices['object_expressions__1'],
                       'pooled_ensemble_adapter':choices['pooled_ensemble'],
                       'clip_prompt_transferred_adapter':choices[source_choice.prompt],
                       'target_selected_adapter':selected,
                       'target_selected_text':text_only,
                    }
                    for method,choice in methods.items():
                        emit('selected_configs.csv',dict(dataset=args.dataset,backbone=model,budget_per_class=budget,
                             repeat=repeat,method=method,**asdict(choice),selection_metric=args.objective,
                             n_unique_labels=len(ids),additional_validation_labels=0))
                        if method!='target_selected_text':oof_guesses=oof[choice.prompt]
                        else:oof_guesses=(100*x[ids]@prototypes[choice.prompt].T).argmax(axis=1)
                        for local,j in enumerate(ids):
                            emit('selected_oof_predictions.csv',dict(dataset=args.dataset,backbone=model,
                                 budget_per_class=budget,repeat=repeat,method=method,path=portable.path.iloc[j],
                                 inner_fold=int(fold_ids[local]),true_label=labels[y[j]],prediction=labels[oof_guesses[local]]))
                    evaluation=[('internal',x[test],y[test],portable.path.iloc[test].tolist())]
                    if external is not None:
                        external_x,_,_=external_bundles[model]
                        external_y=np.array([labels.index(c) for c in external.label])
                        # Image IDs are relative to the external manifest, not personal absolute paths.
                        evaluation.append(('external',external_x,external_y,external_ids))
                    for evaluation_name,queries,truth,image_ids in evaluation:
                        similarities=np.clip(queries@x[ids].T,-1,1)
                        guesses={method:predict_choice(queries,x[ids],y[ids],prototypes,choice,similarities)
                                 for method,choice in methods.items()}
                        guesses['nearest_centroid']=centroid_predict(queries,x[ids],y[ids],len(labels))
                        for method,predicted in guesses.items():
                            emit('metrics_by_run.csv',dict(dataset=args.dataset,evaluation=evaluation_name,
                                 backbone=model,budget_per_class=budget,repeat=repeat,method=method,
                                 n_test=len(truth),n_unique_labels=len(ids),additional_validation_labels=0,
                                 **metrics(truth,predicted,len(labels))))
                            for path,true,guess in zip(image_ids,truth,predicted):
                                emit('predictions.csv',dict(dataset=args.dataset,evaluation=evaluation_name,
                                     backbone=model,budget_per_class=budget,repeat=repeat,method=method,
                                     path=path,true_label=labels[true],prediction=labels[guess],correct=int(true==guess)))
                    print(f'{args.dataset} {model} B={budget} draw {repeat+1}/{args.repeats}: selected {selected.prompt}',flush=True)
                for f in handles:f.flush()
        status='complete'
    except BaseException:
        status='incomplete'
        raise
    finally:
        for f in handles:f.close()
        meta=dict(dataset=args.dataset,seed=args.seed,backbones=args.backbones,budgets=args.budgets,
             repeats=args.repeats,primary_metric=args.objective,primary_comparison='target_selected_adapter minus pooled_ensemble_adapter',
             status=locals().get('status','incomplete'),classes=labels,prompt_sha256=prompt_sha,
             prompt_source='external frozen CSV' if args.prompt_csv else 'historical authored grid; retrospective benchmark extension',
             n_training_pool=len(train),unused_validation_images=len(val),n_test=len(test),
             additional_validation_labels=0,folds=4,final_cache='All B labeled images per class after inner-CV selection',
             transfer='CLIP selects prompt ID using the same labels; target checkpoint re-tunes cache settings in the same pool',
             text_logit_scale=100,feature_metadata={m:bundles[m][2] for m in bundles},
             external_test=external is not None,external_n=0 if external is None else len(external),
             group_aware=('group_id' in portable),
             sampling='One image per sampled group per class' if 'group_id' in portable else 'Class-balanced image sampling',
             benchmark_budget_assumption='Benchmark labels form balanced episodes; this does not model discovering classes in an unlabeled stream',
             caveat='No new-data confirmation or duplicate-free claim is implied by running this protocol')
        (out/'metadata.json').write_text(json.dumps(meta,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--dataset',required=True,choices=['realwaste','trashnet'])
    p.add_argument('--out',required=True);p.add_argument('--features',default='budget_features')
    p.add_argument('--backbones',nargs='+',choices=['clip','siglip','openclip'],default=['clip','siglip','openclip'])
    p.add_argument('--budgets',type=int,nargs='+',default=[4,8,16,32]);p.add_argument('--repeats',type=int,default=30)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--objective',choices=['macro_f1','accuracy'],default='macro_f1')
    p.add_argument('--reference-split',default='');p.add_argument('--split-manifest',default='')
    p.add_argument('--prompt-csv',default='');p.add_argument('--external-manifest',default='');p.add_argument('--batch',type=int,default=16)
    args=p.parse_args()
    if args.split_manifest and args.reference_split:p.error('Use either a historical reference check or a custom split, not both')
    if 'clip' not in args.backbones:p.error('Include clip to retain source-selection audit records')
    if args.repeats<1 or len(set(args.budgets))!=len(args.budgets) or min(args.budgets)<4 or len(set(args.backbones))!=len(args.backbones):p.error('Invalid repetitions, budgets or checkpoint list')
    run(args)
