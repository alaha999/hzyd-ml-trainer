import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


def performance(y, score, signed_weights, threshold=0.5):
    w = np.abs(signed_weights)
    out = {'events': int(len(y)),
           'signal_events': int((y == 1).sum()), 'background_events': int((y == 0).sum()),
           'sumw_signal': float(signed_weights[y == 1].sum()),
           'sumw_background': float(signed_weights[y == 0].sum()),
           'sumw2_signal': float(np.square(signed_weights[y == 1]).sum()),
           'sumw2_background': float(np.square(signed_weights[y == 0]).sum()),
           'negative_events': int((signed_weights < 0).sum()),
           'accuracy': float(accuracy_score(y, score >= threshold))}
    for name, sw in [('auc', None), ('auc_weighted_abs', w)]:
        out[name] = float(roc_auc_score(y, score, sample_weight=sw)) if len(np.unique(y)) == 2 and (sw is None or all(sw[y == c].sum() > 0 for c in (0, 1))) else None
    return out


def scan_significance(y, score, signed_weights, thresholds=None, frac_bkg_systematic=0.0):
    """Simple S/sqrt(B + (f B)^2), signed expected yields, invalid B -> NaN."""
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)
    s, b, z = [], [], []
    for cut in thresholds:
        selected = score >= cut
        signal = signed_weights[selected & (y == 1)].sum()
        background = signed_weights[selected & (y == 0)].sum()
        variance = background + (frac_bkg_systematic * background)**2
        significance = signal / np.sqrt(variance) if signal >= 0 and variance > 0 else np.nan
        s.append(signal); b.append(background); z.append(significance)
    return np.asarray(thresholds), np.asarray(s), np.asarray(b), np.asarray(z)
