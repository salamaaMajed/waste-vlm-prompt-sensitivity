import numpy as np
import pandas as pd
from wastevlm.core import split,normalize,cache_logits,make_class_prototypes,metrics,validate_reference_split
from wastevlm.prompts import prompt_sets


def test_prompt_grid():
    sets=prompt_sets(['cardboard','plastic'])
    assert len(sets)==9
    assert set(k[0] for k in sets)=={'name_preserving','object_expressions','material_framing'}
    assert all(len(x['plastic'])==3 for x in sets.values())
    assert sets[('object_expressions',1)]['plastic'][0]=='a photo of plastic containers and packaging'


def test_split_determinism():
    df=pd.DataFrame({'path':[f'/tmp/{i:03}.jpg' for i in range(100)],'label':['cardboard']*50+['plastic']*50})
    a,y,tr,v,t=split(df,['cardboard','plastic'],42)
    b,*_=split(df,['cardboard','plastic'],42)
    assert a.equals(b)
    assert (len(tr),len(v),len(t))==(60,20,20)
    assert not(set(tr)&set(v) or set(tr)&set(t) or set(v)&set(t))


def test_adapter_kernel_and_prototypes():
    x=np.eye(2,dtype='float32')
    logits=cache_logits(x,x,np.array([0,1]),10,2)
    assert logits.shape==(2,2)
    assert np.array_equal(logits.argmax(1),[0,1])
    prototypes=make_class_prototypes(np.array([[2.,0.],[3.,0.],[0.,2.],[0.,3.]],dtype='float32'),[2,2])
    assert np.allclose(prototypes,x)
    assert metrics(np.array([0,1]),np.array([0,1]),2)['accuracy']==1


def test_reference_prompt_matches():
    import pathlib
    root=pathlib.Path(__file__).resolve().parents[1]
    import pandas as pd
    for dataset,labels in [('realwaste',sorted(set(__import__('wastevlm.prompts',fromlist=['DESCRIPTIONS']).DESCRIPTIONS)-{'trash'})),('trashnet',['cardboard','glass','metal','paper','plastic','trash'])]:
        expected=pd.read_csv(root/'reference'/'zero_shot'/dataset/'clip'/'all_prompts.csv')
        generated=[]
        for (family,variant),mapping in prompt_sets(labels).items():
            for c in labels:
                for i,p in enumerate(mapping[c]):
                    generated.append((family,variant,c,i,p))
        got=pd.DataFrame(generated,columns=['family','variant','category','prompt_index','prompt'])
        assert got.equals(expected)


def test_original_linguistic_prompts():
    from wastevlm.controls import linguistic_prompts,experiment8_prompts
    labels=['cardboard','glass','metal','paper','plastic','trash']
    p=linguistic_prompts(labels)
    assert p['alternate_category_phrases']['plastic'][0]=='a photo of a used plastic bottle'
    assert p['class_name_preserving_templates']['plastic'][1]=='an image of discarded plastic material'
    q=experiment8_prompts(labels,42)
    assert len(q)==5
    assert all(len(prompts)==3 for mapping in q.values() for prompts in mapping.values())
    assert q['shuffled_attributes_3']['plastic'] != q['attributes_3']['plastic']
