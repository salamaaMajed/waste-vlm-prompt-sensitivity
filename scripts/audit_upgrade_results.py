from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import accuracy_score,f1_score

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--results',default='measurements/seed_42')
parser.add_argument('--reference',default='reference')
parser.add_argument('--out',default='verification')
args=parser.parse_args()
ROOT=Path(args.results)
REF=Path(args.reference)
OUT=Path(args.out);OUT.mkdir(parents=True,exist_ok=True)
checks=[];comparisons=[];selection=[];transitions=[];classrec=[]

def check(condition,label):
 if not condition:raise AssertionError(label)
 checks.append(label)

for dataset in ['realwaste','trashnet']:
 print('Checking',dataset,flush=True)
 truth=None;prompts=None;split=None
 for model in ['clip','siglip','openclip']:
  base=ROOT/f'zero_shot/{dataset}/{model}'
  p=pd.read_csv(base/'predictions.csv')
  m=pd.read_csv(base/'split_manifest.csv')
  pr=pd.read_csv(base/'all_prompts.csv')
  expected=pd.read_csv(REF/f'splits/{dataset}.csv')
  key=['path','label','split']
  check(m[key].equals(expected[key]),f'{dataset}/{model}: exact historical split')
  check(not p.duplicated(['path','family','variant']).any(),f'{dataset}/{model}: unique prediction keys')
  check(set(p.path)==set(m[m.split=='test'].path),f'{dataset}/{model}: complete held-out image inventory')
  label_map=dict(zip(m.path,m.label))
  check(all(label_map[path]==label for path,label in zip(p.path,p.true_label)),f'{dataset}/{model}: true labels match manifest')
  check(set(p.groupby('path').size())=={9},f'{dataset}/{model}: all nine configurations on each image')
  check((p.correct==(p.true_label==p.predicted_label).astype(int)).all(),f'{dataset}/{model}: saved correctness flags')
  if prompts is None:prompts=pr;split=m
  else:
   check(pr.equals(prompts),f'{dataset}/{model}: identical prompts across checkpoints')
   check(m.equals(split),f'{dataset}/{model}: identical manifests across checkpoints')
  metrics=[]
  for (family,variant),g in p.groupby(['family','variant']):
   metrics.append(dict(family=family,variant=variant,accuracy=accuracy_score(g.true_label,g.predicted_label),
      macro_f1=f1_score(g.true_label,g.predicted_label,labels=sorted(m.label.unique()),average='macro',zero_division=0)))
  metrics=pd.DataFrame(metrics).sort_values(['family','variant']).reset_index(drop=True)
  saved=pd.read_csv(base/'metrics_by_prompt_set.csv').sort_values(['family','variant']).reset_index(drop=True)
  check(np.allclose(metrics[['accuracy','macro_f1']],saved[['accuracy','macro_f1']],atol=1e-12),f'{dataset}/{model}: per-set metrics reconstructed')
  historical=pd.read_csv(REF/f'zero_shot/{dataset}/{model}/metrics_by_prompt_set.csv').sort_values(['family','variant']).reset_index(drop=True)
  check(np.allclose(metrics[['accuracy','macro_f1']],historical[['accuracy','macro_f1']],atol=1e-12),f'{dataset}/{model}: historical per-set results reproduced')
  g=p[p.family=='name_preserving'].set_index(['variant','path']).sort_index()
  h=p[p.family=='object_expressions'].set_index(['variant','path']).sort_index()
  transitions.append(dict(dataset=dataset,model=model,fixed=int(((g.correct==0)&(h.correct==1)).sum()),
                         harmed=int(((g.correct==1)&(h.correct==0)).sum())))
  for (family,label),data in p[p.family.isin(['name_preserving','object_expressions'])].groupby(['family','true_label']):
   classrec.append(dict(dataset=dataset,model=model,family=family,label=label,recall=100*data.correct.mean()))
  validation=pd.read_csv(base/'validation_metrics.csv')
  selected=pd.read_csv(base/'validation_selected_results.csv')
  for row in selected.itertuples():
   winner=validation.sort_values([row.selection_metric,'family','variant'],ascending=[False,True,True]).iloc[0]
   check(row.family==winner.family and row.variant==winner.variant,f'{dataset}/{model}: {row.selection_metric} choice matches validation rule')
   test=metrics[(metrics.family==row.family)&(metrics.variant==row.variant)].iloc[0]
   check(np.allclose([test.accuracy,test.macro_f1],[row.test_accuracy,row.test_macro_f1]),f'{dataset}/{model}: selected test metrics')
   selection.append(dict(dataset=dataset,model=model,selection_metric=row.selection_metric,family=row.family,
                    variant=row.variant,validation_score=100*row.validation_score,
                    test_accuracy=100*test.accuracy,test_macro_f1=100*test.macro_f1))
  # Recreate bootstrap accuracy intervals via per-image effect vectors, independently
  # of the distributed confusion-matrix/bootstrap implementation.
  wide=p.pivot(index='path',columns=['family','variant'],values='correct').sort_index()
  delta=wide['object_expressions'].mean(axis=1)-wide['name_preserving'].mean(axis=1)
  if truth is None:truth=delta.index;draws=np.random.default_rng(20261017).integers(0,len(delta),size=(5000,len(delta)));effects={};samples={}
  check(delta.index.equals(truth),f'{dataset}/{model}: common paired bootstrap image order')
  effects[model]=100*delta.mean();samples[model]=100*delta.to_numpy()[draws].mean(axis=1)
  saved_ci=pd.read_csv(ROOT/f'paired_analysis/{dataset}/paired_image_intervals.csv')
  row=saved_ci[(saved_ci.comparison==model)&(saved_ci.metric=='accuracy')].iloc[0]
  check(np.allclose([effects[model],*np.quantile(samples[model],[.025,.975])],[row.estimate_pp,row.ci95_low_pp,row.ci95_high_pp],atol=1e-10),f'{dataset}/{model}: image-bootstrap accuracy CI independently reproduced')
 for i,a in enumerate(['clip','siglip','openclip']):
  for b in ['clip','siglip','openclip'][i+1:]:
   row=saved_ci[(saved_ci.comparison==f'{a} minus {b}')&(saved_ci.metric=='accuracy')].iloc[0]
   check(np.allclose([effects[a]-effects[b],*np.quantile(samples[a]-samples[b],[.025,.975])],[row.estimate_pp,row.ci95_low_pp,row.ci95_high_pp],atol=1e-10),f'{dataset}: {a}/{b} accuracy interaction interval')

 baseline_supports=None;baseline_parameters=None;baseline_preds=None
 for mode in ['baseline','independent']:
  base=ROOT/f'few_shot/{dataset}/clip/{mode}'
  p=pd.read_csv(base/'predictions.csv');s=pd.read_csv(base/'support_set_ids.csv');m=pd.read_csv(base/'split_manifest.csv')
  hyper=pd.read_csv(base/'matched_hyperparameters.csv')
  check(m[['path','label','split']].equals(split[['path','label','split']]),f'{dataset}/{mode}: adapter split matches zero-shot')
  partition=dict(zip(m.path,m.split))
  check(all(partition[x]=='train' for x in s.path),f'{dataset}/{mode}: all supports in training pool')
  check(not s.duplicated(['shots','repeat','path']).any(),f'{dataset}/{mode}: no duplicate supports within a draw')
  counts=s.groupby(['shots','repeat','label']).size()
  check(all(n==key[0] for key,n in counts.items()),f'{dataset}/{mode}: correct shot count for every class and draw')
  check(len(s.groupby(['shots','repeat']))==150,f'{dataset}/{mode}: 150 support draws')
  check(set(p.path)==set(m[m.split=='test'].path),f'{dataset}/{mode}: held-out test coverage')
  check((p.correct==(p.true_label==p.prediction).astype(int)).all(),f'{dataset}/{mode}: correctness flags')
  check(all(label_map[path]==label for path,label in zip(p.path,p.true_label)),f'{dataset}/{mode}: truth matches image manifest')
  metrics=[]
  for (method,shots,repeat),g in p.groupby(['method','shots','repeat']):
   check(len(g)==len(truth) and g.path.nunique()==len(truth),f'{dataset}/{mode}/{method}/{shots}/{repeat}: complete unique test predictions')
   metrics.append(dict(method=method,shots=shots,repeat=repeat,accuracy=accuracy_score(g.true_label,g.prediction),
     macro_f1=f1_score(g.true_label,g.prediction,labels=sorted(m.label.unique()),average='macro',zero_division=0)))
  metrics=pd.DataFrame(metrics)
  saved=pd.read_csv(base/'metrics_by_run.csv')
  merged=metrics.merge(saved,on=['method','shots','repeat'],suffixes=('_calc','_saved'),validate='one_to_one')
  check(len(merged)==302,f'{dataset}/{mode}: all per-run records')
  for metric in ['accuracy','macro_f1']:
   check(np.allclose(merged[f'{metric}_calc'],merged[f'{metric}_saved'],atol=1e-12),f'{dataset}/{mode}: {metric} from predictions')
  if mode=='baseline':
   baseline_supports=s;baseline_parameters=hyper;baseline_preds=p
   check(np.array_equal(hyper.beta,hyper.alternative_beta) and np.array_equal(hyper.alpha,hyper.alternative_alpha),f'{dataset}: shared-tuning parameters identical')
  else:
   check(s.equals(baseline_supports),f'{dataset}: supports identical across tuning protocols')
   check(np.array_equal(hyper.beta,baseline_parameters.beta) and np.array_equal(hyper.alpha,baseline_parameters.alpha),f'{dataset}: baseline tuning identical across protocols')
   aa=baseline_preds[baseline_preds.method=='tip_adapter_matched'].sort_values(['shots','repeat','path'])
   bb=p[p.method=='tip_adapter_independently_tuned'].sort_values(['shots','repeat','path'])
   check(np.array_equal(aa.prediction.to_numpy(),bb.prediction.to_numpy()),f'{dataset}: baseline predictions identical across tuning protocols')
  names=('tip_adapter_matched','alternative_expression_tip_adapter_matched') if mode=='baseline' else ('tip_adapter_independently_tuned','alternative_expression_tip_adapter_independently_tuned')
  saved_ci=pd.read_csv(base/'paired_draw_intervals.csv')
  for metric in ['accuracy','macro_f1']:
   for shots in [1,2,4,8,16]:
    grid=metrics[metrics.shots==shots].pivot(index='repeat',columns='method',values=metric)
    diff=100*(grid[names[1]]-grid[names[0]])
    half=t.ppf(.975,29)*diff.std(ddof=1)/np.sqrt(30)
    row=saved_ci[(saved_ci.shots==shots)&(saved_ci.metric==metric)].iloc[0]
    check(np.allclose([diff.mean(),diff.mean()-half,diff.mean()+half],[row.mean_gain_pp,row.ci95_low_pp,row.ci95_high_pp],atol=1e-10),f'{dataset}/{mode}/{metric}/{shots}: paired draw interval')
    comparisons.append(dict(dataset=dataset,tuning_mode=mode,metric=metric,shots=shots,
         name_mean=100*grid[names[0]].mean(),alternative_mean=100*grid[names[1]].mean(),gain_pp=diff.mean(),
         ci95_low_pp=diff.mean()-half,ci95_high_pp=diff.mean()+half))

pd.DataFrame(comparisons).to_csv(OUT/'verified_adapter_comparisons.csv',index=False)
pd.DataFrame(selection).to_csv(OUT/'verified_validation_selection.csv',index=False)
pd.DataFrame(transitions).to_csv(OUT/'verified_error_transitions.csv',index=False)
pd.DataFrame(classrec).to_csv(OUT/'verified_class_recall.csv',index=False)
(OUT/'audit_status.json').write_text(json.dumps(dict(checks_passed=len(checks),checks=checks,
 limits=['Image pixels, model weights and validation-grid predictions not included; inference and the hyperparameter optimum cannot be independently regenerated here.',
        'Single fixed split; authored prompt vocabulary unchanged; no new independent dataset or duplicate-group-clean evaluation.']),indent=2))
print('PASS',len(checks),'checks; all numerical reconstruction and pairing checks passed')
