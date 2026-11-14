#!/usr/bin/env python3
"""Build matched-template prompts from actual, completed human-author forms."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import pandas as pd
from wastevlm.core import classes_for

FAMILIES=['name_preserving','material_framing','object_expressions']
TEMPLATES=['a photo of {description}','an image of {description}','a close-up of {description}']


def build(form,out):
    forms=[form] if isinstance(form,(str,Path)) else list(form)
    frame=pd.concat([pd.read_csv(p) for p in forms],ignore_index=True)
    keys=['author_id','category','family','description_index']
    if not set(keys+['description']).issubset(frame):raise ValueError('Incomplete author form columns')
    if frame[keys+['description']].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError('Descriptions must be genuinely completed; blanks or duplicate entries found')
    if not frame.description.map(lambda x:isinstance(x,str) and bool(x.strip())).all():raise ValueError('Blank description')
    authors=sorted(frame.author_id.unique())
    if len(authors)!=3 or set(frame.family)!=set(FAMILIES):raise ValueError('Expected three authors and three prescribed families')
    all_labels=sorted(set(classes_for('realwaste'))|set(classes_for('trashnet')))
    expected={(a,c,f,i) for a in authors for c in all_labels for f in FAMILIES for i in (1,2,3)}
    if set(map(tuple,frame[keys].itertuples(index=False,name=None)))!=expected:
        raise ValueError('Expected three descriptions per author/category/family across the complete inventory')
    for row in frame[frame.family=='name_preserving'].itertuples():
        if not re.search(r'(?<!\w)'+re.escape(row.category)+r'(?!\w)',row.description,re.IGNORECASE):
            raise ValueError(f'{row.author_id}/{row.category}: name-preserving phrase must retain the exact category name')
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new output directory to preserve frozen prompts')
    out.mkdir(parents=True,exist_ok=True)
    language_audit=frame.copy()
    language_audit['word_count']=language_audit.description.str.split().map(len)
    language_audit['character_count']=language_audit.description.str.len()
    language_audit.to_csv(out/'description_audit.csv',index=False)
    for dataset in ['realwaste','trashnet']:
        rows=[]
        for variant,author in enumerate(authors,1):
            for family in FAMILIES:
                for category in classes_for(dataset):
                    values=frame[(frame.author_id==author)&(frame.category==category)&(frame.family==family)].sort_values('description_index')
                    phrases=values.description.str.strip().tolist()
                    for i,(description,template) in enumerate((d,t) for d in phrases for t in TEMPLATES):
                        rows.append(dict(family=family,variant=variant,category=category,prompt_index=i,prompt=template.format(description=description)))
        pd.DataFrame(rows).to_csv(out/f'{dataset}_prompts.csv',index=False)
    digests=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in forms]
    (out/'provenance.json').write_text(json.dumps(dict(author_ids=authors,author_form_sha256=digests,
        templates=TEMPLATES,prompts_per_class_per_candidate=9,
        note='The program validates completeness, not human independence. Keep consent and authorship records separately.'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--form',required=True,nargs='+');p.add_argument('--out',default='independent_prompts')
    args=p.parse_args();build(args.form,args.out)
