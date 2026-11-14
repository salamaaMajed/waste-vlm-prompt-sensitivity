"""Prepare content-bound feature bundles; model imports stay lazy for CPU audits."""
from pathlib import Path
import gc
import hashlib
import json
import numpy as np
from .prompts import prompt_sets


def prompt_inventory(labels, prompt_csv=''):
    if prompt_csv:
        from .custom_prompts import read_prompt_csv
        groups=read_prompt_csv(prompt_csv,labels)
    else:
        groups=prompt_sets(labels)
    mappings={f'{family}__{variant}':mapping for (family,variant),mapping in groups.items()}
    if 'name_preserving__1' not in mappings or 'object_expressions__1' not in mappings:
        raise ValueError('The fixed baseline IDs name_preserving__1 and object_expressions__1 are required')
    candidates=sorted(mappings)
    mappings['class_name']={c:[f'a photo of {c} waste'] for c in labels}
    mappings['pooled_ensemble']={c:sorted({p for key in candidates for p in mappings[key][c]}) for c in labels}
    serialized=json.dumps(dict(labels=labels,mappings=mappings),sort_keys=True,ensure_ascii=False)
    digest=hashlib.sha256(serialized.encode()).hexdigest()
    return mappings,candidates,digest


def image_fingerprint(paths):
    # File-byte hashes prevent a same-path image edit from silently reusing features.
    digest=hashlib.sha256()
    for path in paths:
        path=Path(path)
        digest.update(str(path.resolve()).encode())
        with path.open('rb') as f:
            while True:
                block=f.read(1024*1024)
                if not block:break
                digest.update(block)
    return digest.hexdigest()


def feature_bundle(paths, labels, mappings, prompt_sha, backbone, folder, fingerprint, batch=16):
    path=Path(folder)/f'{backbone}.npz'
    expected=dict(image_fingerprint=fingerprint,prompt_sha256=prompt_sha,backbone=backbone,classes=labels)
    if path.is_file():
        with np.load(path,allow_pickle=False) as z:
            meta=json.loads(str(z['metadata'].item()))
            if all(meta.get(k)==v for k,v in expected.items()) and np.array_equal(z['paths'],np.asarray(paths,dtype=str)):
                prototypes={key:z['prototypes'][i] for i,key in enumerate(z['prompt_ids'].tolist())}
                return z['features'].copy(),prototypes,meta
        raise ValueError(f'{path}: stale feature bundle; use a new feature folder')
    from .encoders import Encoder,CHECKPOINTS
    from .env import environment
    encoder=Encoder(backbone)
    x=encoder.encode_images(paths,batch=batch)
    prototypes={key:encoder.prototypes(mapping,labels) for key,mapping in sorted(mappings.items())}
    cfg=getattr(encoder.model,'config',None)
    meta=dict(**expected,checkpoint=CHECKPOINTS[backbone],environment=environment(),
              resolved_revision=getattr(cfg,'_commit_hash',None),
              revision_note='Null means the backend did not expose a resolved revision; no bitwise weight guarantee')
    Path(folder).mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,features=x,paths=np.asarray(paths,dtype=str),
                        prototypes=np.stack(list(prototypes.values())),prompt_ids=np.asarray(list(prototypes),dtype=str),
                        metadata=np.array(json.dumps(meta)))
    del encoder
    gc.collect()
    import torch
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return x,prototypes,meta
