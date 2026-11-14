"""Synthetic integration checks; no model, network or real image inference.

These test protocol implementation only, never the paper's numerical accuracy.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import runpy
import sys
import numpy as np
import pandas as pd
from wastevlm.core import classes_for, normalize


class FakeEncoder:
    def __init__(self,family):
        self.family=family
    def encode_images(self,paths,batch=16):
        labels=classes_for('trashnet')
        rng=np.random.default_rng(19)
        base=np.stack([np.eye(len(labels))[labels.index(Path(p).parent.name)] for p in paths])
        return normalize(base+0.8*rng.normal(size=base.shape))
    def prototypes(self,mapping,labels):
        basic=all(mapping[c]==[f'a photo of {c} waste'] for c in labels)
        rng=np.random.default_rng(17 if basic else 18)
        return normalize(np.eye(len(labels))+0.5*rng.normal(size=(len(labels),len(labels))))


def test_fewshot_tuning_modes_share_supports_and_use_separate_ids():
    root=Path(__file__).resolve().parents[1]
    # The actual backend is deliberately unavailable to this test.
    with patch.dict(sys.modules,{'torch':SimpleNamespace(manual_seed=lambda seed:None)}):
        ns=runpy.run_path(str(root/'scripts/run_few_shot.py'))
        ns['run'].__globals__['Encoder']=FakeEncoder
        with TemporaryDirectory() as d:
            d=Path(d)
            for label in classes_for('trashnet'):
                (d/'data'/label).mkdir(parents=True)
                for i in range(20):(d/'data'/label/f'{i:03}.jpg').touch()
            for mode in ['baseline','independent']:
                args=SimpleNamespace(seed=42,data=str(d/'data'),dataset='trashnet',out=str(d/mode),
                     backbone='clip',tuning_mode=mode,shots=[1,2],repeats=2,reference_split='',cache_dir='',batch=16)
                ns['run'](args)
            a=pd.read_csv(d/'baseline/support_set_ids.csv')
            b=pd.read_csv(d/'independent/support_set_ids.csv')
            assert a.equals(b)
            params=pd.read_csv(d/'baseline/matched_hyperparameters.csv')
            assert (params.beta==params.alternative_beta).all()
            assert (params.alpha==params.alternative_alpha).all()
            params=pd.read_csv(d/'independent/matched_hyperparameters.csv')
            assert params.alternative_validation_accuracy.notna().all()
            methods=set(pd.read_csv(d/'independent/metrics_by_run.csv').method)
            assert 'tip_adapter_independently_tuned' in methods
            assert 'tip_adapter_matched' not in methods


def test_zero_shot_runner_exports_validation_selection():
    root=Path(__file__).resolve().parents[1]
    with patch.dict(sys.modules,{'torch':SimpleNamespace(manual_seed=lambda seed:None)}):
        ns=runpy.run_path(str(root/'scripts/run_zero_shot.py'))
        ns['run'].__globals__['Encoder']=FakeEncoder
        with TemporaryDirectory() as d:
            d=Path(d)
            for label in classes_for('trashnet'):
                (d/'data'/label).mkdir(parents=True)
                for i in range(20):(d/'data'/label/f'{i:03}.jpg').touch()
            args=SimpleNamespace(seed=42,data=str(d/'data'),dataset='trashnet',out=str(d/'out'),
                backbone='clip',reference_split='',cache_dir='',batch=16,prompt_csv='')
            ns['run'](args)
            val=pd.read_csv(d/'out/validation_metrics.csv')
            selected=pd.read_csv(d/'out/validation_selected_results.csv')
            for row in selected.itertuples():
                winner=val.sort_values([row.selection_metric,'family','variant'],ascending=[False,True,True]).iloc[0]
                assert row.family==winner.family and row.variant==winner.variant
