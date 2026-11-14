"""Verbatim linguistic controls from original Experiments 8 and follow-up."""
import numpy as np
from .prompts import DESCRIPTIONS

ATTRIBUTES = {
'cardboard':['corrugated brown fiberboard packaging','a folded carton with layered paper fibers','rigid paperboard boxes and packaging'],
'glass':['transparent brittle glass bottles and jars','glass containers with reflective hard surfaces','colored glass fragments or containers'],
'metal':['reflective aluminum or steel containers','metal cans with rigid metallic surfaces','shiny metal scraps and packaging'],
'paper':['thin flexible paper sheets and documents','newspapers with printed text on paper','flat fibrous paper sheets'],
'plastic':['molded synthetic plastic bottles and containers','flexible plastic film wrappers and bags','polymer packaging with smooth surfaces'],
'trash':['mixed discarded objects and residual refuse','assorted general garbage not sorted by material','mixed nonrecyclable household refuse'],
'food organics':['discarded fruit and vegetable food scraps','leftover organic food and kitchen waste','food remains and peels from meals'],
'miscellaneous trash':['mixed discarded objects and residual refuse','assorted general garbage not sorted by material','mixed nonrecyclable household refuse'],
'textile trash':['woven fabrics and discarded garments','cloth scraps with visible fabric fibers','used clothing and textile waste'],
'vegetation':['green leaves and garden trimmings','cut grass and discarded plant matter','branches and other garden vegetation']}
ALTERNATIVES = {
'cardboard':['a used corrugated box','discarded carton packaging','waste made from thick layered paperboard'],
'glass':['a used glass bottle','discarded glass jars','waste objects made of glass'],
'metal':['a used metal can','discarded metallic packaging','waste objects made from metal'],
'paper':['a discarded paper document','used newspaper sheets','waste paper pages'],
'plastic':['a used plastic bottle','discarded polymer containers','waste plastic packaging'],
'trash':['assorted unsorted rubbish','general residual garbage','mixed discarded refuse'],
'food organics':['kitchen food leftovers','discarded edible remains','food scraps from cooking'],
'miscellaneous trash':['assorted unsorted rubbish','general residual garbage','mixed discarded refuse'],
'textile trash':['discarded old clothing','used fabric scraps','waste textile items'],
'vegetation':['garden leaf litter','discarded plant cuttings','yard grass trimmings']}


def experiment8_prompts(labels,seed=42):
    groups={
        'class_name_3':{c:[f'a photo of {c} waste']*3 for c in labels},
        'generic_3':{c:['a photograph of discarded household waste','an image of an unwanted item','a picture of a waste material'] for c in labels},
        'attributes_3':{c:[f'a photo of {v}' for v in ATTRIBUTES[c]] for c in labels},
        'paraphrases_3':{c:[f'a photo of {v}' for v in ALTERNATIVES[c]] for c in labels},
    }
    rng=np.random.default_rng(seed)
    perm=np.arange(len(labels))
    for _ in range(10000):
        rng.shuffle(perm)
        if np.all(perm!=np.arange(len(labels))):break
    else:raise RuntimeError('Could not make derangement')
    groups['shuffled_attributes_3']={c:[f'a photo of {v}' for v in ATTRIBUTES[labels[j]]] for c,j in zip(labels,perm)}
    return groups


def linguistic_prompts(labels):
    return {
        'repeated_class_name':{c:[f'a photo of {c} waste']*3 for c in labels},
        'class_name_preserving_templates':{c:[f'a photo of {c} waste',f'an image of discarded {c} material',f'a picture showing {c} refuse'] for c in labels},
        'alternate_category_phrases':{c:[f'a photo of {v}' for v in ALTERNATIVES[c]] for c in labels},
    }
