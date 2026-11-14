from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,f1_score
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--results',default='measurements/seed_42')
parser.add_argument('--out',default='verification')
args=parser.parse_args()
r=Path(args.results);out=Path(args.out);out.mkdir(parents=True,exist_ok=True);records=[]
for dataset in ['realwaste','trashnet']:
 clip_selection=pd.read_csv(r/f'zero_shot/{dataset}/clip/validation_selected_results.csv')
 source_choice=clip_selection[clip_selection.selection_metric=='accuracy'].iloc[0]
 for model in ['clip','siglip','openclip']:
  base=r/f'zero_shot/{dataset}/{model}'
  p=pd.read_csv(base/'predictions.csv');chosen=pd.read_csv(base/'validation_selected_results.csv')
  own=chosen[chosen.selection_metric=='accuracy'].iloc[0]
  target=p[(p.family==own.family)&(p.variant==own.variant)].sort_values('path')
  y=target.true_label.to_numpy();yp=target.predicted_label.to_numpy()
  rng=np.random.default_rng(20261017);idx=rng.integers(0,len(y),size=(5000,len(y)))
  for comparison,family,variant in [('fixed_name_v1','name_preserving',1),('clip_validation_choice',source_choice.family,source_choice.variant)]:
   comp=p[(p.family==family)&(p.variant==variant)].sort_values('path')
   assert np.array_equal(target.path.to_numpy(),comp.path.to_numpy())
   other=comp.predicted_label.to_numpy()
   diff=(yp==y).astype(float)-(other==y).astype(float)
   ci=100*np.quantile(diff[idx].mean(axis=1),[.025,.975])
   records.append(dict(dataset=dataset,model=model,selection_metric='accuracy',selected_family=own.family,
                       selected_variant=int(own.variant),comparison=comparison,reference_family=family,
                       reference_variant=int(variant),selected_accuracy=100*accuracy_score(y,yp),
                       reference_accuracy=100*accuracy_score(y,other),gain_pp=100*diff.mean(),
                       ci95_low_pp=ci[0],ci95_high_pp=ci[1],
                       scope='Exploratory paired image interval; fixed validation choices and prompts; unadjusted'))
pd.DataFrame(records).to_csv(out/'selection_transfer_comparisons.csv',index=False)
print(pd.DataFrame(records).to_string(index=False))
