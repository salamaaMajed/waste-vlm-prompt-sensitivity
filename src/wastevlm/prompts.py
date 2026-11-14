"""Verbatim Experiment 10 category descriptions and prompt families."""
DESCRIPTIONS = {
 'cardboard':['corrugated cardboard packaging','a brown cardboard box or sheet','paperboard packaging material'],
 'glass':['glass bottles or broken glass','transparent or colored glass material','glass container fragments'],
 'metal':['metal cans and metallic scrap','aluminum or steel waste material','shiny metallic objects or containers'],
 'paper':['paper sheets and documents','printed paper and newspapers','thin paper waste material'],
 'plastic':['plastic containers and packaging','synthetic plastic waste items','plastic bottles bags and wrappers'],
 'trash':['mixed non recyclable garbage','assorted general household refuse','unidentified mixed residual waste'],
 'food organics':['discarded food scraps','leftover food and kitchen organic waste','fruit vegetable or food remains'],
 'miscellaneous trash':['mixed non recyclable garbage','assorted general household refuse','unidentified mixed residual waste'],
 'textile trash':['discarded fabric and clothing','textile scraps and used garments','cloth fibers and fabric waste'],
 'vegetation':['green garden vegetation waste','leaves grass and plant trimmings','plant matter and garden debris']}

TEMPLATES = [
  ["a photo of {c} waste", "a discarded piece of {c} material", "an image showing used {c}"],
  ["a photograph showing {c}", "an example of discarded {c}", "a close-up of waste made of {c}"],
  ["waste category: {c}", "material category: {c}", "a household waste image labeled {c}"],
]
OBJECT_TEMPLATES = [
  ["a photo of {d0}", "a photo of {d1}", "a photo of {d2}"],
  ["discarded item: {d1}", "a waste object such as {d2}", "used material: {d0}"],
  ["an image of {d2}", "a close-up of {d0}", "household discard: {d1}"],
]
MATERIAL_TEMPLATES = [
  ["material waste: {c}", "discarded {c} matter", "a sample of {c} refuse"],
  ["recyclable or discarded {c} material", "the material {c} in waste", "waste composed of {c}"],
  ["a material classification photograph of {c}", "used {c} material", "an object made from {c}"],
]

def prompt_sets(labels):
    groups={}
    for family, variants in [('name_preserving',TEMPLATES),('object_expressions',OBJECT_TEMPLATES),('material_framing',MATERIAL_TEMPLATES)]:
        for v,templates in enumerate(variants,1):
            mapping={}
            for c in labels:
                d=DESCRIPTIONS[c]
                mapping[c]=[t.format(c=c,d0=d[0],d1=d[1],d2=d[2]) for t in templates]
            groups[(family,v)]=mapping
    return groups
