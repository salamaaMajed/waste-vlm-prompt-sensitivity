"""Label-budget selection with cross-validation confined to the sampled pool.

No test labels or historical validation labels are accepted by the selector.
All methods reuse the same sampled labels and cross-validation folds.
"""
from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold
from .core import normalize


@dataclass(frozen=True)
class Choice:
    prompt: str
    beta: float
    alpha: float
    cv_accuracy: float
    cv_macro_f1: float


def budget_ids(train, y, budget, repeat, seed=42, groups=None):
    if budget < 4:
        raise ValueError('At least four labels per class are required by this protocol')
    rng=np.random.default_rng(seed+1009*repeat+97*budget)
    ids=[]
    for c in sorted(np.unique(y)):
        pool=np.asarray(train)[y[train]==c]
        if len(pool)<budget:
            raise ValueError(f'Class {c}: fewer than {budget} training-pool images')
        if groups is None:
            ids.extend(rng.choice(pool,size=budget,replace=False).tolist())
        else:
            groups=np.asarray(groups)
            available=np.unique(groups[pool])
            if len(available)<budget:raise ValueError(f'Class {c}: fewer than B distinct source groups')
            selected_groups=rng.choice(available,size=budget,replace=False)
            ids.extend([int(rng.choice(pool[groups[pool]==g])) for g in selected_groups])
    return np.asarray(ids,dtype=int)


def inner_folds(y, seed=42, groups=None):
    counts=np.bincount(y)
    if len(counts)<2 or counts.min()<4:
        raise ValueError('Every class needs at least four labeled examples')
    if groups is None:
        return list(StratifiedKFold(n_splits=4,shuffle=True,random_state=seed).split(np.zeros(len(y)),y))
    if any(len(np.unique(np.asarray(groups)[y==c]))<4 for c in np.unique(y)):
        raise ValueError('Each class needs four distinct groups for group-aware inner CV; deduplicate or revise grouping')
    return list(StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=seed).split(np.zeros(len(y)),y,groups))


def score_guesses(y, guessed, nclasses):
    """Vectorized accuracy and macro-F1 for [configuration, image] predictions."""
    guessed=np.atleast_2d(guessed)
    codes=y[None,:]*nclasses+guessed
    offset=np.arange(len(guessed))[:,None]*nclasses*nclasses
    cm=np.bincount((codes+offset).ravel(),minlength=len(guessed)*nclasses*nclasses).reshape(-1,nclasses,nclasses)
    diag=np.diagonal(cm,axis1=-2,axis2=-1)
    denom=cm.sum(axis=1)+cm.sum(axis=2)
    f1=np.divide(2*diag,denom,out=np.zeros_like(diag,dtype=float),where=denom!=0).mean(axis=1)
    acc=diag.sum(axis=1)/len(y)
    return np.stack([acc,f1],axis=1)


def cv_search(x, y, prototypes, folds, objective='macro_f1'):
    """Search prompts and cache parameters using only the budgeted labels.

    Folds train on 3/4 of the budget; after selection the final cache uses all
    budgeted examples. Scores pool out-of-fold predictions before macro-F1.
    The fixed scale is 100. Alpha=0 allows a text-only branch.
    Ties: sorted prompt ID, then alpha=0, then beta/positive-alpha grid order.
    """
    if objective not in ('accuracy','macro_f1'):
        raise ValueError(objective)
    nclasses=len(np.unique(y))
    if set(np.unique(y))!=set(range(nclasses)):
        raise ValueError('Labels must be contiguous class indices')
    coverage=np.zeros(len(y),dtype=int)
    cache={beta:np.empty((len(y),nclasses),dtype=np.float32) for beta in (1,5,10,20)}
    for train,valid in folds:
        if set(train)&set(valid) or set(train)|set(valid)!=set(range(len(y))):
            raise ValueError('Invalid inner fold')
        coverage[valid]+=1
        sim=np.clip(x[valid]@x[train].T,-1,1)
        values=np.eye(nclasses,dtype=np.float32)[y[train]]
        for beta in cache:
            cache[beta][valid]=np.exp(np.clip(-beta*(1-sim),-80,0))@values
    if not np.all(coverage==1):
        raise ValueError('Each budgeted image must have exactly one out-of-fold prediction')
    grid=[(1,0)]+[(beta,alpha) for beta in (1,5,10,20) for alpha in (1,5,10,20,50)]
    records=[];choices={};oof={}
    metric_index=0 if objective=='accuracy' else 1
    for prompt in sorted(prototypes):
        text=100*x@prototypes[prompt].T
        guesses=np.stack([(text+alpha*cache[beta]).argmax(axis=1) for beta,alpha in grid])
        scores=score_guesses(y,guesses,nclasses)
        best=int(np.argmax(scores[:,metric_index]))
        beta,alpha=grid[best]
        choices[prompt]=Choice(prompt,beta,alpha,float(scores[best,0]),float(scores[best,1]))
        oof[prompt]=guesses[best]
        for (beta,alpha),score in zip(grid,scores):
            records.append(dict(prompt=prompt,beta=beta,alpha=alpha,cv_accuracy=float(score[0]),
                                cv_macro_f1=float(score[1]),n_cv_labels=len(y)))
    return choices,records,oof


def best_prompt(choices, candidate_ids, objective='macro_f1', text_only=False, records=None):
    candidates=sorted(candidate_ids)
    if not candidates:
        raise ValueError('No prompt candidates')
    if text_only:
        if records is None:raise ValueError('Text-only selection requires all CV records')
        entries=[r for r in records if r['prompt'] in candidates and r['alpha']==0]
        winner=max(entries,key=lambda r:r['cv_'+objective])
        return Choice(winner['prompt'],winner['beta'],0,winner['cv_accuracy'],winner['cv_macro_f1'])
    return max((choices[p] for p in candidates),key=lambda c:getattr(c,'cv_'+objective))


def predict_choice(query, support, support_y, prototypes, choice, similarities=None):
    nclasses=prototypes[choice.prompt].shape[0]
    text=100*query@prototypes[choice.prompt].T
    if choice.alpha==0:return text.argmax(axis=1)
    sim=np.clip(query@support.T if similarities is None else similarities,-1,1)
    kernel=np.exp(np.clip(-choice.beta*(1-sim),-80,0))
    cache=kernel@np.eye(nclasses,dtype=np.float32)[support_y]
    return (text+choice.alpha*cache).argmax(axis=1)


def centroid_predict(query, support, support_y, nclasses):
    centers=normalize(np.stack([support[support_y==c].mean(axis=0) for c in range(nclasses)]))
    return (query@centers.T).argmax(axis=1)
