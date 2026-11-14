import numpy as np
import pandas as pd
from wastevlm.paired_analysis import FAMILIES, align_predictions, bootstrap_family_scores, contrast_rows


def sample_frame():
    rows=[]
    for family in FAMILIES:
        for variant in (1,2,3):
            for i in range(12):
                rows.append(dict(path=f'{i}.jpg', family=family, variant=variant,
                                 true_label=str(i%2), predicted_label=str(i%2)))
    return pd.DataFrame(rows)


def test_identical_checkpoints_have_exact_zero_interaction():
    models,labels,variants,paths,y,p=align_predictions({'a':sample_frame(),'b':sample_frame()})
    observed,boot=bootstrap_family_scores(y,p,len(labels),100,42)
    rows=contrast_rows(models,observed,boot,'toy')
    assert np.allclose(rows[['estimate_pp','ci95_low_pp','ci95_high_pp']],0)


def test_bootstrap_preserves_variant_image_dependence():
    # Perfectly correlated variants must give the same CI as one variant.
    y=np.array([0,1]*10)
    pred=np.zeros((1,3,1,20),dtype=int)
    pred[:]=y
    pred[0,2,0,:5]=1-y[:5]
    a,ba=bootstrap_family_scores(y,pred,2,100,19)
    b,bb=bootstrap_family_scores(y,np.repeat(pred,3,axis=2),2,100,19)
    assert np.allclose(a,b) and np.allclose(ba,bb)


def test_alignment_rejects_missing_prediction_and_conflicting_truth():
    for changed in (sample_frame().iloc[:-1], sample_frame().assign(true_label='wrong')):
        try:
            align_predictions({'a':sample_frame(),'b':changed})
        except ValueError:
            pass
        else:
            raise AssertionError('Mismatched prediction evidence was accepted')


def test_macro_f1_matches_sklearn():
    from sklearn.metrics import f1_score
    y=np.array([0,0,0,1,1,2])
    guess=np.array([0,1,0,0,1,0])
    pred=np.broadcast_to(guess,(1,3,3,6))
    scores,_=bootstrap_family_scores(y,pred,3,10,1)
    assert np.allclose(scores[...,1],f1_score(y,guess,labels=[0,1,2],average='macro',zero_division=0))
