"""Safe, checkpoint-tagged cache for local image features."""
from pathlib import Path
import numpy as np
from .encoders import CHECKPOINTS

def load_or_encode(encoder, paths, output, batch=16):
    Path(output).mkdir(parents=True, exist_ok=True)
    cache=Path(output)/f'image_features_{encoder.family}.npz'
    key='|'.join(str(x) for x in CHECKPOINTS[encoder.family] if x is not None)
    paths=np.asarray(paths,dtype=str)
    if cache.exists():
        with np.load(cache,allow_pickle=False) as z:
            if np.array_equal(z['paths'],paths) and str(z['checkpoint'].item())==key:
                print(f'Using verified image cache {cache}',flush=True)
                return z['features']
    features=encoder.encode_images(paths.tolist(),batch=batch)
    np.savez_compressed(cache,paths=paths,checkpoint=np.array(key),features=features)
    return features
