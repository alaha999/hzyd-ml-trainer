"""Preprocessing and weight handling shared by both training backends."""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def apply_negative_weight_policy(x, y, w, process, drop_negative_weight=False, absolute_negative_weight=False):
    if drop_negative_weight and absolute_negative_weight:
        raise ValueError('Select only one: drop_negative_weight or absolute_negative_weight')
    if not np.isfinite(x).all() or not np.isfinite(w).all():
        raise ValueError('Nonfinite feature or weight detected')
    neg = w < 0
    print(f'[WEIGHTS] {neg.sum():,} negative-weight entries')
    if drop_negative_weight:
        keep = ~neg
        return x[keep], y[keep], w[keep], process[keep]
    # Preserve signed weights for yields; abs weights are used for training/ML metrics.
    if neg.any() and not absolute_negative_weight:
        raise ValueError('Negative MC weights exist; select drop_negative_weight: true or absolute_negative_weight: true')
    return x, y, w, process

def split_scale(x, y, w, process, test_fraction=0.2, seed=42):
    itr, ite = train_test_split(np.arange(len(y)), test_size=test_fraction, stratify=y, random_state=seed)
    scaler = StandardScaler()
    xtr = scaler.fit_transform(x[itr]).astype('float32')
    xte = scaler.transform(x[ite]).astype('float32')
    return itr, ite, scaler, xtr, xte

def balance_training_weights(w, y, use_weights=True, balanced_sample_weight=True):
    weights = np.abs(w).astype(np.float64) if use_weights else np.ones(len(w), dtype=np.float64)
    if balanced_sample_weight:
        for cls in (0, 1):
            sel = y == cls
            total = weights[sel].sum()
            if total <= 0: raise ValueError(f'Class {cls} has zero training weight')
            weights[sel] /= total
    if weights.sum() <= 0: raise ValueError('Total training weight is zero')
    weights *= len(weights) / weights.sum()
    return weights.astype('float32')
