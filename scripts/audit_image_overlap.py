#!/usr/bin/env python3
"""Global byte/pixel duplicate and coarse perceptual-neighbor screen.

dHash neighbors are review candidates, never automatically confirmed duplicates.
All image pairs are screened, including pairs within an existing partition.
"""
import argparse
import csv
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image,ImageOps


def run(data,manifest,out,threshold=4,external_manifest=''):
    if not 0<=threshold<=64:raise ValueError('dHash threshold must be 0..64')
    source=pd.read_csv(manifest)
    if not {'path','label','split'}.issubset(source):raise ValueError('Source manifest requires path,label,split')
    source['image_path']=source.path.map(lambda p:str((Path(data)/str(p)).resolve()))
    source['image_id']=['source_'+str(i) for i in range(len(source))]
    if external_manifest:
        ext=pd.read_csv(external_manifest)
        if not {'path','label','source_id'}.issubset(ext):raise ValueError('External manifest requires source_id provenance')
        ext['image_path']=ext.path.map(lambda p:str((Path(external_manifest).resolve().parent/str(p)).resolve()))
        ext['image_id']=['external_'+str(i) for i in range(len(ext))];ext['split']='external'
        source=pd.concat([source,ext],ignore_index=True)
    rows=[]
    for record in source.to_dict('records'):
        path=Path(record['image_path'])
        file_hash=hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as image:
            rgb=ImageOps.exif_transpose(image).convert('RGB')
            pixel_hash=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
            gray=np.asarray(rgb.convert('L').resize((9,8),Image.Resampling.LANCZOS))
            bits=(gray[:,1:]>gray[:,:-1]).ravel()
            dhash=int.from_bytes(np.packbits(bits).tobytes(),'big')
        rows.append(dict(image_id=record['image_id'],path=record['path'],label=record['label'],split=record['split'],
             source_id=record.get('source_id',''),known_group=record.get('group_id',''),
             sha256_file=file_hash,sha256_pixels=pixel_hash,dhash_hex=f'{dhash:016x}'))
    frame=pd.DataFrame(rows);out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new audit output directory')
    out.mkdir(parents=True,exist_ok=True);frame.to_csv(out/'hash_manifest.csv',index=False)
    codes=np.array([int(v,16) for v in frame.dhash_hex],dtype=np.uint64)
    lookup=np.array([i.bit_count() for i in range(256)],dtype=np.uint8)
    count=0
    with (out/'candidate_pairs.csv').open('w',newline='') as f:
        fields=['image_id_a','path_a','split_a','label_a','image_id_b','path_b','split_b','label_b',
                'kind','dhash_distance','cross_partition','human_confirmed_duplicate','review_note']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for i in range(len(frame)):
            xor=np.ascontiguousarray(codes[i+1:]^codes[i])
            distance=lookup[xor.view(np.uint8).reshape(-1,8)].sum(axis=1)
            for j in np.flatnonzero(distance<=threshold)+i+1:
                a,b=frame.iloc[i],frame.iloc[j]
                kind=('exact_file' if a.sha256_file==b.sha256_file else
                      'exact_pixels' if a.sha256_pixels==b.sha256_pixels else 'perceptual_candidate')
                writer.writerow(dict(image_id_a=a.image_id,path_a=a.path,split_a=a.split,label_a=a.label,
                    image_id_b=b.image_id,path_b=b.path,split_b=b.split,label_b=b.label,kind=kind,
                    dhash_distance=int(distance[j-i-1]),cross_partition=bool(a.split!=b.split),
                    human_confirmed_duplicate='',review_note=''))
                count+=1
    (out/'SCOPE.txt').write_text(
       f'Globally screened {len(frame)} image files; {count} exact/perceptual candidate pairs.\n'
       'Exact pixel hashes use EXIF-corrected RGB pixels. dHash is a coarse screen, not a complete near-duplicate detector.\n'
       'Review candidates using actual images. Absence of candidates does not establish absence of related scenes.\n'
       'External source independence must also be established from collection provenance.\n')
    print(f'Wrote {len(frame)} image hashes and {count} candidate pairs')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--out',required=True);p.add_argument('--threshold',type=int,default=4);p.add_argument('--external-manifest',default='')
    a=p.parse_args();run(a.data,a.manifest,a.out,a.threshold,a.external_manifest)
