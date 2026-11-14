"""Small synthetic feature-bundle integration test; no research measurements."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import runpy
import numpy as np
import pandas as pd
from wastevlm.core import discover,normalize
from wastevlm.budget_features import prompt_inventory,image_fingerprint


def test_budget_runner_and_summary_on_synthetic_cached_features():
    root=Path(__file__).resolve().parents[1]
    runner=runpy.run_path(str(root/'scripts/run_budget_experiment.py'))['run']
    summarizer=runpy.run_path(str(root/'scripts/summarize_budget_experiment.py'))['summarize']
    auditor=runpy.run_path(str(root/'scripts/audit_budget_results.py'))['audit']
    feature_auditor=runpy.run_path(str(root/'scripts/audit_budget_features.py'))['audit']
    with TemporaryDirectory() as d:
        d=Path(d)
        for label in ['cardboard','glass','metal','paper','plastic','trash']:
            folder=d/'data'/label;folder.mkdir(parents=True)
            for i in range(20):(folder/f'{i:03}.jpg').touch()
        df,labels=discover(d/'data','trashnet')
        mappings,candidates,digest=prompt_inventory(labels)
        fingerprint=image_fingerprint(df.path.tolist())
        features=d/'features/trashnet/source';features.mkdir(parents=True)
        for model in ['clip','siglip']:
            rng=np.random.default_rng(23 if model=='clip' else 29)
            y=np.array([labels.index(c) for c in df.label])
            x=normalize(np.eye(6)[y]+.6*rng.normal(size=(len(df),6)))
            proto={key:normalize(np.eye(6)+.4*rng.normal(size=(6,6))) for key in sorted(mappings)}
            meta=dict(image_fingerprint=fingerprint,prompt_sha256=digest,backbone=model,classes=labels,backend='synthetic_test')
            np.savez_compressed(features/f'{model}.npz',features=x,paths=np.asarray(df.path,dtype=str),
              prototypes=np.stack(list(proto.values())),prompt_ids=np.asarray(list(proto),dtype=str),metadata=np.array(json.dumps(meta)))
        args=SimpleNamespace(data=str(d/'data'),dataset='trashnet',out=str(d/'results/trashnet'),features=str(d/'features'),
            backbones=['clip','siglip'],budgets=[4],repeats=2,seed=42,objective='macro_f1',reference_split='',
            split_manifest='',prompt_csv='',external_manifest='',batch=16)
        runner(args)
        pools=pd.read_csv(d/'results/trashnet/labeled_pool.csv')
        selected=pd.read_csv(d/'results/trashnet/selected_configs.csv')
        scores=pd.read_csv(d/'results/trashnet/cv_scores.csv')
        assert (pools.groupby('repeat').size()==24).all()
        assert (selected.additional_validation_labels==0).all()
        assert (selected.n_unique_labels==24).all()
        assert len(pd.read_csv(d/'results/trashnet/metrics_by_run.csv'))==32
        # Reconstruct target choices from every recorded inner-CV candidate score.
        for (model,repeat),g in scores.groupby(['backbone','repeat']):
            candidate=g[g.prompt.isin(candidates)]
            maximum=candidate.cv_macro_f1.max()
            row=selected[(selected.backbone==model)&(selected.repeat==repeat)&(selected.method=='target_selected_adapter')].iloc[0]
            assert np.isclose(row.cv_macro_f1,maximum)
        summarizer(d/'results',d/'summary')
        auditor(d/'results/trashnet')
        exported=d/'exported/trashnet';exported.mkdir(parents=True)
        for model in ['clip','siglip']:
            with np.load(features/f'{model}.npz',allow_pickle=False) as z:
                arrays={key:z[key].copy() for key in z.files}
            arrays['paths']=np.asarray([Path(p).relative_to(d/'data').as_posix() for p in arrays['paths']],dtype=str)
            np.savez_compressed(exported/f'{model}.npz',**arrays)
        feature_auditor(d/'results/trashnet',d/'exported',first_draw_only=False)
        contrasts=pd.read_csv(d/'summary/paired_comparisons.csv')
        assert contrasts.primary.sum()==2
        assert contrasts[contrasts.primary].primary_holm_p.notna().all()
        predictions=pd.read_csv(d/'results/trashnet/predictions.csv')
        a=predictions[(predictions.backbone=='clip')&(predictions.method=='target_selected_adapter')].sort_values(['repeat','path'])
        b=predictions[(predictions.backbone=='clip')&(predictions.method=='clip_prompt_transferred_adapter')].sort_values(['repeat','path'])
        assert np.array_equal(a.prediction.to_numpy(),b.prediction.to_numpy())
        try:runner(args)
        except ValueError:pass
        else:raise AssertionError('Existing measurements were overwritten')
