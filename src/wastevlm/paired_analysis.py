"""Paired image bootstrap; prompt variants stay attached to their source image.

Intervals condition on the fixed authored prompts, checkpoint weights and split.
They do not estimate uncertainty over authors, prompt populations or datasets.
"""
import numpy as np
import pandas as pd

FAMILIES = ('name_preserving', 'material_framing', 'object_expressions')


def align_predictions(frames):
    """Fail rather than silently intersect mismatched or incomplete experiments."""
    models = list(frames)
    arrays = []
    reference = None
    variants = None
    paths = None
    truth = None
    labels = None
    for model, frame in frames.items():
        required = {'path', 'family', 'variant', 'true_label', 'predicted_label'}
        if not required.issubset(frame.columns):
            raise ValueError(f'{model}: missing columns {required - set(frame.columns)}')
        frame = frame.copy()
        if frame[list(required)].isna().any().any():
            raise ValueError(f'{model}: null prediction field')
        if frame.duplicated(['path', 'family', 'variant']).any():
            raise ValueError(f'{model}: duplicate prediction key')
        if set(frame.family) != set(FAMILIES):
            raise ValueError(f'{model}: expected exactly the three historical families')
        if (frame.groupby('path').true_label.nunique() != 1).any():
            raise ValueError(f'{model}: inconsistent true labels')
        current_variants = sorted(frame.variant.unique())
        current_paths = sorted(frame.path.unique())
        expected = pd.MultiIndex.from_product(
            [FAMILIES, current_variants, current_paths], names=['family', 'variant', 'path'])
        keyed = frame.set_index(['family', 'variant', 'path'])
        if len(keyed) != len(expected) or set(keyed.index) != set(expected):
            raise ValueError(f'{model}: incomplete image/family/variant grid')
        keyed = keyed.reindex(expected)
        current_truth = frame.groupby('path').true_label.first().reindex(current_paths)
        if reference is None:
            reference = expected
            variants, paths = current_variants, current_paths
            truth = current_truth
            labels = sorted(truth.unique())
        elif (not expected.equals(reference) or not current_truth.equals(truth)):
            raise ValueError(f'{model}: images, labels or variants differ across checkpoints')
        lookup = {c: i for i, c in enumerate(labels)}
        if not set(keyed.predicted_label).issubset(lookup):
            raise ValueError(f'{model}: unknown predicted label')
        arrays.append(keyed.predicted_label.map(lookup).to_numpy().reshape(
            len(FAMILIES), len(variants), len(paths)))
    if not arrays:
        raise ValueError('No prediction frames supplied')
    return models, labels, variants, paths, truth.map({c:i for i,c in enumerate(labels)}).to_numpy(), np.stack(arrays)


def bootstrap_family_scores(y, predictions, n_classes, n_resamples=5000, seed=20261017):
    """Return observed and bootstrapped mean-of-variant accuracy and macro-F1.

    predictions: [model, family, variant, image]. Each bootstrap draw samples
    image IDs once and uses those same IDs for every variant and checkpoint.
    Macro-F1 is recomputed within each variant before the variant mean.
    """
    if n_resamples < 2:
        raise ValueError('At least two bootstrap resamples are required')
    n = len(y)
    if predictions.shape[-1] != n or n == 0:
        raise ValueError('Prediction/image count mismatch')
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_resamples, n))
    shape = predictions.shape[:-1]
    observed = np.empty(shape + (2,))
    boot = np.empty(shape + (n_resamples, 2))
    for index in np.ndindex(shape):
        guessed = predictions[index]
        codes = y * n_classes + guessed
        cm = np.bincount(codes, minlength=n_classes**2).reshape(n_classes, n_classes)
        observed[index] = _scores(cm)
        for start in range(0, n_resamples, 128):
            chosen = draws[start:start + 128]
            offset = np.arange(len(chosen))[:, None] * n_classes**2
            cms = np.bincount((codes[chosen] + offset).ravel(),
                              minlength=len(chosen) * n_classes**2).reshape(-1, n_classes, n_classes)
            boot[index][start:start+len(chosen)] = _scores(cms)
    # Average metric values, not predictions or confusion matrices, over sets.
    return observed.mean(axis=2), boot.mean(axis=2)


def _scores(cm):
    diagonal = np.diagonal(cm, axis1=-2, axis2=-1)
    denominator = cm.sum(axis=-1) + cm.sum(axis=-2)
    f1 = np.divide(2 * diagonal, denominator, out=np.zeros_like(diagonal, dtype=float),
                   where=denominator != 0).mean(axis=-1)
    accuracy = diagonal.sum(axis=-1) / cm.sum(axis=(-2, -1))
    return np.stack([accuracy, f1], axis=-1)


def contrast_rows(models, observed, boot, dataset):
    """Conditional percentile CIs; no uncorrected significance declarations."""
    name = FAMILIES.index('name_preserving')
    obj = FAMILIES.index('object_expressions')
    effects = observed[:, obj] - observed[:, name]
    sampled = boot[:, obj] - boot[:, name]
    rows = []
    def append(label, kind, values, samples):
        for j, metric in enumerate(('accuracy', 'macro_f1')):
            lo, hi = np.quantile(samples[:, j], [.025, .975]) * 100
            rows.append(dict(dataset=dataset, comparison=label, kind=kind, metric=metric,
                             estimate_pp=100*values[j], ci95_low_pp=lo, ci95_high_pp=hi,
                             n_resamples=samples.shape[0],
                             scope='Paired image bootstrap; fixed prompts/checkpoints/split; unadjusted'))
    for i, model in enumerate(models):
        append(model, 'object_minus_name', effects[i], sampled[i])
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            append(f'{models[i]} minus {models[j]}', 'checkpoint_interaction',
                   effects[i]-effects[j], sampled[i]-sampled[j])
    return pd.DataFrame(rows)
