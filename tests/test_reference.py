from pathlib import Path
import pandas as pd
from wastevlm.core import classes_for


def test_archived_summaries():
    root=Path(__file__).resolve().parents[1]/'reference'
    for dataset in ('realwaste','trashnet'):
        for backbone in ('clip','siglip','openclip'):
            f=root/'zero_shot'/dataset/backbone/'family_summary.csv'
            df=pd.read_csv(f)
            assert len(df)==3
            assert set(df.family)=={'name_preserving','material_framing','object_expressions'}
            assert (df.sets==3).all()
            assert df.accuracy_mean.between(0,1).all()
        few=pd.read_csv(root/'few_shot'/dataset/'paired_draw_intervals.csv')
        assert len(few)==10
        assert set(few.shots)=={1,2,4,8,16}
        assert (few.n_draws==30).all()
        manifest=pd.read_csv(root/'splits'/f'{dataset}.csv')
        assert set(manifest.split)=={'train','validation','test'}
        assert set(manifest.label)==set(classes_for(dataset))


def test_original_support_set_selection():
    import numpy as np
    from sklearn.model_selection import train_test_split
    root=Path(__file__).resolve().parents[1]/'reference'
    for dataset in ('realwaste','trashnet'):
        df=pd.read_csv(root/'splits'/f'{dataset}.csv')
        supports=pd.read_csv(root/'few_shot'/dataset/'support_set_ids.csv')
        names=sorted(df.label.unique());y=np.array([names.index(c) for c in df.label]);ix=np.arange(len(df))
        train,_=train_test_split(ix,test_size=.4,stratify=y,random_state=42)
        for shots in (1,2,4,8,16):
            for rep in range(30):
                rng=np.random.default_rng(42+1009*rep+97*shots)
                chosen=np.concatenate([rng.choice(train[y[train]==c],size=shots,replace=False) for c in range(len(names))])
                expected=supports[(supports.shots==shots)&(supports.repeat==rep)].path.tolist()
                assert df.path.iloc[chosen].tolist()==expected
