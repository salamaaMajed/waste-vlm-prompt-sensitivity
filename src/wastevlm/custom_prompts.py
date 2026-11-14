"""Strict import of frozen externally authored treatments, without test tuning."""
import pandas as pd


def read_prompt_csv(path, labels):
    frame=pd.read_csv(path)
    keys=['family','variant','category','prompt_index']
    if not set(keys+['prompt']).issubset(frame):
        raise ValueError('Prompt CSV requires family, variant, category, prompt_index, prompt')
    if frame[keys+['prompt']].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError('Missing field or duplicate prompt key')
    if not frame.prompt.map(lambda p:isinstance(p,str) and bool(p.strip())).all():
        raise ValueError('Empty prompt')
    for column in ['variant','prompt_index']:
        if not pd.to_numeric(frame[column],errors='coerce').map(lambda v:pd.notna(v) and float(v).is_integer()).all():
            raise ValueError(f'{column} must contain integers')
        frame[column]=frame[column].astype(int)
    groups={}
    counts=set()
    family_variants={}
    for (family,variant),g in frame.groupby(['family','variant'],sort=True):
        if set(g.category)!=set(labels):
            raise ValueError(f'{family}/{variant}: class inventory mismatch')
        mapping={c:g[g.category==c].sort_values('prompt_index').prompt.tolist() for c in labels}
        lengths={len(v) for v in mapping.values()}
        if len(lengths)!=1:
            raise ValueError('All classes must have equal prompt count')
        counts.update(lengths)
        family_variants.setdefault(family,set()).add(variant)
        groups[(family,variant)]=mapping
    if not groups or len(counts)!=1 or len({tuple(sorted(v)) for v in family_variants.values()})!=1:
        raise ValueError('Balanced prompt counts and matched variants required across families')
    return groups
