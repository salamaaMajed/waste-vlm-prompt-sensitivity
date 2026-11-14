#!/usr/bin/env python3
"""Build 3/1/1 five-fold group partitions after image-overlap review."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def build(hashes,out,reviewed='',seed=42):
    full=pd.read_csv(hashes)
    source=full[full.image_id.str.startswith('source_')].copy().reset_index(drop=True)
    parent={key:key for key in full.image_id}
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for col in ['sha256_pixels','known_group']:
        if col not in full:continue
        valid=full[full[col].notna() & full[col].astype(str).str.strip().ne('')]
        for _,g in valid.groupby(col):
            ids=g.image_id.tolist()
            for key in ids[1:]:union(ids[0],key)
    if reviewed:
        pairs=pd.read_csv(reviewed)
        flags=pairs.human_confirmed_duplicate.astype(str).str.lower().str.strip()
        if not flags.isin(['true','false']).all():raise ValueError('Every supplied review row needs TRUE or FALSE')
        for row in pairs[flags=='true'].itertuples():
            if row.image_id_a not in parent or row.image_id_b not in parent:raise ValueError('Unknown image ID in manual review')
            union(row.image_id_a,row.image_id_b)
    full['group_id']=full.image_id.map(find)
    # A purported external test must not contain a confirmed source duplicate.
    for _,g in full.groupby('group_id'):
        if g.image_id.str.startswith('source_').any() and g.image_id.str.startswith('external_').any():
            raise ValueError('Confirmed source/external overlap: remove or recollect external examples before evaluation')
    source['group_id']=source.image_id.map(find)
    folds=np.full(len(source),-1)
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
    for i,(_,holdout) in enumerate(splitter.split(np.zeros(len(source)),source.label,source.group_id)):folds[holdout]=i
    source['split']=np.where(folds==0,'test',np.where(folds==1,'validation','train'))
    if (source.groupby('group_id').split.nunique()>1).any():raise AssertionError('Group leakage')
    for split in ['train','validation','test']:
        if set(source[source.split==split].label)!=set(source.label):raise ValueError('A group split omitted a class; inspect grouping or use another documented split seed')
    out=Path(out)
    if out.exists():raise ValueError('Refusing to overwrite a frozen group split')
    out.parent.mkdir(parents=True,exist_ok=True)
    source[['path','label','split','group_id']].to_csv(out,index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--hashes',required=True);p.add_argument('--out',required=True)
    p.add_argument('--reviewed-candidates',default='');p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();build(a.hashes,a.out,a.reviewed_candidates,a.seed)
