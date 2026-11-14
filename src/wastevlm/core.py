"""Dataset discovery, deterministic split, and exact Experiment 6-12 scoring conventions."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from .prompts import DESCRIPTIONS

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}


def classes_for(dataset):
    if dataset == 'trashnet':
        return sorted({'cardboard','glass','metal','paper','plastic','trash'})
    if dataset == 'realwaste':
        return sorted(set(DESCRIPTIONS) - {'trash'})
    raise ValueError(f'Unknown dataset: {dataset}')


def discover(root, dataset):
    """Reproduce original sorting of absolute paths; class is the parent folder."""
    allowed = set(classes_for(dataset))
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    rows = []
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            label = p.parent.name.lower().replace('_', ' ').strip()
            if label in allowed:
                rows.append((str(p.resolve()), label))
    df = pd.DataFrame(rows, columns=['path', 'label']).sort_values('path').reset_index(drop=True)
    missing = allowed - set(df.label)
    if missing:
        raise ValueError(f'Missing classes {sorted(missing)} under {root}')
    return df, sorted(allowed)


def split(df, labels, seed=42):
    """Exact historical 60/20/20 stratified train/val/test split."""
    y = np.array([labels.index(c) for c in df.label], dtype=int)
    idx = np.arange(len(df))
    train, rest = train_test_split(idx, test_size=.4, stratify=y, random_state=seed)
    val, test = train_test_split(rest, test_size=.5, stratify=y[rest], random_state=seed)
    manifest = df.copy()
    manifest['split'] = 'train'
    manifest.loc[val, 'split'] = 'validation'
    manifest.loc[test, 'split'] = 'test'
    return manifest, y, train, val, test


def relative_manifest(manifest, root):
    root = Path(root).resolve()
    portable = manifest.copy()
    portable['path'] = portable.path.map(lambda p: Path(p).relative_to(root).as_posix())
    return portable


def validate_reference_split(manifest, root, reference_csv):
    if not reference_csv:
        return
    ref = pd.read_csv(reference_csv)
    def key(p):
        parts = str(p).replace('\\', '/').split('/')
        return tuple(parts[-2:])
    actual = {key(p): s for p, s in zip(manifest.path, manifest.split)}
    expected = {key(p): s for p, s in zip(ref.path, ref.split)}
    if len(actual) != len(manifest) or actual != expected:
        raise ValueError('Current split differs from the paper reference split. Check dataset version, paths and seed.')


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def metrics(y_true, y_pred, n_classes):
    return dict(accuracy=float(accuracy_score(y_true, y_pred)),
                macro_f1=float(f1_score(y_true, y_pred, labels=np.arange(n_classes),
                                         average='macro', zero_division=0)))


def cache_logits(query, support, support_y, beta, nclasses):
    """Original Tip-Adapter kernel: exp(-beta * (1 - cosine))."""
    onehot = np.eye(nclasses, dtype='float32')[support_y]
    parts = []
    for start in range(0, len(query), 512):
        sim = query[start:start+512] @ support.T
        parts.append(np.exp(np.clip(-beta*(1-sim), -80, 0)) @ onehot)
    return np.concatenate(parts)


def make_class_prototypes(embeddings, group_sizes):
    """Normalize each text vector before mean, then normalize the class mean."""
    embeddings = normalize(embeddings)
    out = []
    start = 0
    for size in group_sizes:
        out.append(embeddings[start:start+size].mean(0))
        start += size
    return normalize(np.stack(out).astype('float32'))
