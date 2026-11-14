"""Three exact checkpoint families used in the paper, without finetuning."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .core import normalize, make_class_prototypes

CHECKPOINTS = {
    'clip': ('openai/clip-vit-base-patch16', None),
    'siglip': ('google/siglip-base-patch16-224', None),
    'openclip': ('ViT-B-16', 'laion2b_s34b_b88k'),
}

class Encoder:
    def __init__(self, family, device=None):
        self.family = family
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        if family == 'openclip':
            import open_clip
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(*CHECKPOINTS[family])
            self.tokenizer = open_clip.get_tokenizer(CHECKPOINTS[family][0])
        elif family in ('clip','siglip'):
            from transformers import CLIPModel, CLIPProcessor, SiglipModel, SiglipProcessor
            model_cls, proc_cls = (CLIPModel, CLIPProcessor) if family == 'clip' else (SiglipModel, SiglipProcessor)
            self.model = model_cls.from_pretrained(CHECKPOINTS[family][0])
            self.processor = proc_cls.from_pretrained(CHECKPOINTS[family][0])
        else:
            raise ValueError(family)
        self.model = self.model.to(self.device).eval()

    def encode_images(self, paths, batch=16):
        chunks=[]
        for start in range(0, len(paths), batch):
            if self.family == 'openclip':
                ims=[]
                for p in paths[start:start+batch]:
                    with Image.open(p) as im: ims.append(self.preprocess(im.convert('RGB')))
                inputs=torch.stack(ims).to(self.device)
                with torch.inference_mode(): z=self.model.encode_image(inputs)
            else:
                ims=[]
                for p in paths[start:start+batch]:
                    with Image.open(p) as im: ims.append(im.convert('RGB'))
                inputs=self.processor(images=ims,return_tensors='pt').to(self.device)
                with torch.inference_mode(): z=self.model.get_image_features(**inputs)
                if not isinstance(z, torch.Tensor): z=z.pooler_output
            chunks.append(torch.nn.functional.normalize(z.float(),dim=-1).cpu().numpy())
            if start % (batch*25) == 0:
                print(f'Encoded {min(start+batch,len(paths))}/{len(paths)} images', flush=True)
        return np.concatenate(chunks).astype('float32')

    def encode_texts(self, texts, batch=32):
        chunks=[]
        for start in range(0,len(texts),batch):
            ts=texts[start:start+batch]
            if self.family=='openclip':
                tokens=self.tokenizer(ts).to(self.device)
                with torch.inference_mode(): z=self.model.encode_text(tokens)
            else:
                kw={'padding':'max_length'} if self.family=='siglip' else {'padding':True}
                inputs=self.processor(text=ts,truncation=True,return_tensors='pt',**kw).to(self.device)
                with torch.inference_mode(): z=self.model.get_text_features(**inputs)
                if not isinstance(z,torch.Tensor): z=z.pooler_output
            chunks.append(torch.nn.functional.normalize(z.float(),dim=-1).cpu().numpy())
        return np.concatenate(chunks).astype('float32')

    def prototypes(self, mapping, labels):
        prompts=[p for c in labels for p in mapping[c]]
        return make_class_prototypes(self.encode_texts(prompts),[len(mapping[c]) for c in labels])
