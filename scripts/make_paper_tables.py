#!/usr/bin/env python3
"""Aggregate actual run outputs into paper-ready CSVs and PNG figures."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',required=True,help='Directory containing zero_shot/<dataset>/<backbone> and few_shot/<dataset>')
    p.add_argument('--out',default='paper_outputs')
    args=p.parse_args()
    root=Path(args.results);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    zs=[];fs=[]
    for dataset in ('realwaste','trashnet'):
        for model in ('clip','siglip','openclip'):
            f=root/'zero_shot'/dataset/model/'family_summary.csv'
            if f.exists():
                df=pd.read_csv(f);df.insert(0,'dataset',dataset);df.insert(1,'backbone',model);zs.append(df)
        f=root/'few_shot'/dataset/'paired_draw_intervals.csv'
        if f.exists():fs.append(pd.read_csv(f))
    if zs:
        z=pd.concat(zs,ignore_index=True)
        z.to_csv(out/'table_cross_backbone.csv',index=False)
        for dataset in z.dataset.unique():
            d=z[z.dataset==dataset]
            fig,ax=plt.subplots(figsize=(9,4.5))
            names=['name_preserving','material_framing','object_expressions']
            for model in ('clip','siglip','openclip'):
                sub=d[d.backbone==model].set_index('family')
                if len(sub):ax.plot(names,[sub.loc[x,'accuracy_mean']*100 for x in names],marker='o',label=model)
            ax.set_ylabel('Mean accuracy across 3 authored prompt sets (%)')
            ax.set_xlabel('Linguistic prompt family');ax.set_title(dataset)
            ax.legend();fig.tight_layout();fig.savefig(out/f'figure_prompt_families_{dataset}.png',dpi=180);plt.close(fig)
    if fs:
        f=pd.concat(fs,ignore_index=True)
        f.to_csv(out/'table_fewshot_paired.csv',index=False)
        fig,ax=plt.subplots(figsize=(8,4.5))
        for dataset in ('realwaste','trashnet'):
            d=f[(f.dataset==dataset)&(f.metric=='accuracy')].sort_values('shots')
            if len(d):
                ax.errorbar(d.shots,d.mean_gain_pp,yerr=[d.mean_gain_pp-d.ci95_low_pp,d.ci95_high_pp-d.mean_gain_pp],marker='o',capsize=3,label=dataset)
        ax.axhline(0,color='black',linewidth=.8)
        ax.set_xlabel('Shots per class');ax.set_ylabel('AE-Tip minus Tip-Adapter accuracy (pp)')
        ax.set_title('Unadjusted 95% t intervals across support draws (fixed test set)')
        ax.legend();fig.tight_layout();fig.savefig(out/'figure_fewshot_paired.png',dpi=180);plt.close(fig)
    print('Generated:',*[str(x) for x in out.iterdir()],sep='\n - ')

if __name__=='__main__':main()
