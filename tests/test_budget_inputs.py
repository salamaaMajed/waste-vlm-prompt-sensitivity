from pathlib import Path
from tempfile import TemporaryDirectory
import runpy
import numpy as np
import pandas as pd
from PIL import Image
from wastevlm.core import classes_for


def test_independent_author_form_rejects_blanks_and_builds_equal_template_sets():
    root=Path(__file__).resolve().parents[1]
    build=runpy.run_path(str(root/'scripts/build_independent_prompts.py'))['build']
    labels=sorted(set(classes_for('realwaste'))|set(classes_for('trashnet')))
    rows=[dict(author_id=a,category=c,family=f,description_index=i,description=f'toy test phrase {a} {c} {f} {i}')
          for a in ['a','b','c'] for c in labels for f in ['name_preserving','material_framing','object_expressions'] for i in [1,2,3]]
    with TemporaryDirectory() as d:
        d=Path(d);form=d/'form.csv';frame=pd.DataFrame(rows);frame.to_csv(form,index=False)
        build(form,d/'complete')
        p=pd.read_csv(d/'complete/trashnet_prompts.csv')
        assert (p.groupby(['family','variant','category']).size()==9).all()
        assert len(p)==3*3*6*9
        frame.loc[0,'description']='';frame.to_csv(form,index=False)
        try:build(form,d/'blank')
        except ValueError:pass
        else:raise AssertionError('Empty form was represented as completed human work')


def test_pixel_duplicates_and_external_overlap_are_identified():
    root=Path(__file__).resolve().parents[1]
    audit=runpy.run_path(str(root/'scripts/audit_image_overlap.py'))['run']
    grouping=runpy.run_path(str(root/'scripts/make_group_split.py'))['build']
    with TemporaryDirectory() as d:
        d=Path(d);image=Image.fromarray(np.arange(12*12*3,dtype=np.uint8).reshape(12,12,3))
        image.save(d/'one.png');image.save(d/'same_pixels.bmp')
        pd.DataFrame([dict(path='one.png',label='paper',split='train')]).to_csv(d/'source.csv',index=False)
        pd.DataFrame([dict(path='same_pixels.bmp',label='paper',source_id='synthetic_fixture')]).to_csv(d/'external.csv',index=False)
        audit(d,d/'source.csv',d/'audit',4,d/'external.csv')
        pairs=pd.read_csv(d/'audit/candidate_pairs.csv')
        assert len(pairs)==1 and pairs.iloc[0].kind=='exact_pixels'
        try:grouping(d/'audit/hash_manifest.csv',d/'split.csv')
        except ValueError as e:assert 'source/external overlap' in str(e)
        else:raise AssertionError('Confirmed external/source pixel overlap was accepted')


def test_group_split_is_reproducible_and_keeps_exact_matches_together():
    root=Path(__file__).resolve().parents[1]
    grouping=runpy.run_path(str(root/'scripts/make_group_split.py'))['build']
    rows=[]
    for c in range(6):
        for group in range(12):
            for variant in range(2):
                i=len(rows)
                rows.append(dict(image_id=f'source_{i}',path=f'class{c}/{group}_{variant}.jpg',label=f'class{c}',
                                 split='train',sha256_pixels=f'pixels_{c}_{group}',known_group=''))
    with TemporaryDirectory() as d:
        d=Path(d);pd.DataFrame(rows).to_csv(d/'hashes.csv',index=False)
        grouping(d/'hashes.csv',d/'a.csv',seed=42);grouping(d/'hashes.csv',d/'b.csv',seed=42)
        a=pd.read_csv(d/'a.csv');b=pd.read_csv(d/'b.csv')
        assert a.equals(b) and set(a.split)=={'train','validation','test'}
        assert (a.groupby('group_id').split.nunique()==1).all()
