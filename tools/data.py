from pathlib import Path
import numpy as np
import uproot


def read_samples(config, features, weight_branch, tree_default):
    """Return X, labels, signed MC weights, and per-event process names."""
    xs, ys, ws, ps = [], [], [], []
    for name, item in config.items():
        path = Path(item['path']).expanduser()
        tree = item.get('tree', tree_default)
        branches = list(dict.fromkeys(features + ([weight_branch] if weight_branch else [])))
        with uproot.open(path) as f:
            arr = f[tree].arrays(branches, library='np')
        x = np.column_stack([arr[v] for v in features]).astype(np.float32)
        for feature in item.get('mev_to_gev', []):
            if feature not in features:
                raise ValueError(f'{name}: conversion feature {feature} not in features')
            x[:, features.index(feature)] /= 1000.0
        w = np.asarray(arr[weight_branch] if weight_branch else np.ones(len(x)), dtype=np.float64)
        y = np.full(len(x), int(item['label']), dtype=np.int64)
        if y[0:1].size and y[0] not in (0, 1):
            raise ValueError(f'{name}: label must be 0 or 1')
        if not (np.isfinite(x).all() and np.isfinite(w).all()):
            raise ValueError(f'{name}: non-finite features or weights')
        xs.append(x); ys.append(y); ws.append(w); ps.append(np.full(len(x), name, dtype=object))
        print(f'[READ] {name:<22} label={item["label"]} entries={len(x):,} sumW={w.sum():.6g}')
    if not xs:
        raise ValueError('No processes configured')
    return tuple(map(np.concatenate, (xs, ys, ws, ps)))

