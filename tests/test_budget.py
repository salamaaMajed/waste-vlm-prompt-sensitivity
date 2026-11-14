import numpy as np
from sklearn.metrics import f1_score,accuracy_score
from wastevlm.budget import budget_ids,inner_folds,cv_search,best_prompt,predict_choice,score_guesses,centroid_predict
from wastevlm.core import normalize


def toy():
    rng=np.random.default_rng(51)
    y=np.repeat(np.arange(3),12)
    x=normalize(np.eye(3)[y]+.4*rng.normal(size=(len(y),3)))
    proto={'name_preserving__1':np.eye(3,dtype=np.float32),'object_expressions__1':normalize(np.roll(np.eye(3),1,axis=0))}
    return x,y,proto


def test_budget_draw_never_uses_heldout_and_is_shared_deterministically():
    _,y,_=toy();train=np.flatnonzero(np.arange(len(y))%3!=0)
    a=budget_ids(train,y,4,0,42);b=budget_ids(train,y,4,0,42)
    assert np.array_equal(a,b) and set(a)<=set(train) and len(np.unique(a))==12
    assert np.array_equal(np.bincount(y[a]),[4,4,4])


def test_cross_validation_excludes_every_query_and_predicts_once():
    x,y,p=toy();ids=budget_ids(np.arange(len(y)),y,4,0);folds=inner_folds(y[ids])
    coverage=np.zeros(len(ids),dtype=int)
    for train,valid in folds:
        assert not set(train)&set(valid)
        coverage[valid]+=1
    assert (coverage==1).all()
    choices,records,oof=cv_search(x[ids],y[ids],p,folds)
    winner=best_prompt(choices,list(p))
    assert len(records)==42 and winner.prompt in p and len(oof[winner.prompt])==len(ids)


def test_vectorized_macro_f1_keeps_absent_classes():
    y=np.array([0,0,1,1]);guesses=np.array([[0,1,1,1],[2,2,2,2]])
    scores=score_guesses(y,guesses,3)
    for i,g in enumerate(guesses):
        assert np.isclose(scores[i,0],accuracy_score(y,g))
        assert np.isclose(scores[i,1],f1_score(y,g,labels=[0,1,2],average='macro',zero_division=0))


def test_fit_has_no_test_label_input_and_prediction_is_deterministic():
    x,y,p=toy();ids=budget_ids(np.arange(len(y)),y,4,0)
    choices,_,_=cv_search(x[ids],y[ids],p,inner_folds(y[ids]))
    choice=best_prompt(choices,list(p))
    a=predict_choice(x,x[ids],y[ids],p,choice)
    b=predict_choice(x,x[ids],y[ids],p,choice)
    assert np.array_equal(a,b)
    assert len(centroid_predict(x,x[ids],y[ids],3))==len(x)


def test_group_folds_do_not_split_duplicate_groups():
    y=np.repeat(np.arange(3),8);groups=np.repeat(np.arange(12),2)
    for train,valid in inner_folds(y,13,groups):
        assert not set(groups[train])&set(groups[valid])


def test_invalid_fold_reuse_is_rejected():
    x,y,p=toy()
    try:cv_search(x,y,p,[(np.arange(len(y)),np.arange(len(y)))])
    except ValueError:pass
    else:raise AssertionError('Self-including validation cache was accepted')
