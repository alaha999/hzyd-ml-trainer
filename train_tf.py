#!/usr/bin/env python3
"""Train the TensorFlow backend; original PyTorch train.py stays unchanged."""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tools.data import read_samples
from tools.preprocess import apply_negative_weight_policy, split_scale, balance_training_weights
from tools.models.loader import load_model_module
from tools.metrics import performance
from tools import plotting as plots


def main():
    parser = argparse.ArgumentParser(description='Train HEP classifier with TensorFlow')
    parser.add_argument('-c', '--config', default='config_tf.yaml')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    general, tr, mc, options = cfg['general'], cfg['training'], cfg['model'], cfg.get('plots', {})
    seed = int(tr.get('seed', 42))
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)

    out = Path(general.get('output_dir', 'output')) / datetime.now().strftime('training_tf_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    features = general['features']
    # Always load signed MC weights for reporting, regardless of training switches.
    x, y, w, process = read_samples(cfg['samples'], features, general.get('weight_branch'), general['tree'])
    x, y, w, process = apply_negative_weight_policy(x, y, w, process, **{k: tr.get(k, False) for k in ('drop_negative_weight', 'absolute_negative_weight')})
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Training requires both labels 0 and 1')
    itr, ite, scaler, xtr, xte = split_scale(x,y,w,process,test_fraction=float(tr.get('test_fraction',.2)),seed=seed)
    ytr, yte = y[itr], y[ite]
    wtr, wte = w[itr], w[ite]
    wt = balance_training_weights(wtr, ytr, tr.get('use_weights', True), tr.get('balanced_sample_weight', True))

    model_module = load_model_module(mc['model_path'])
    model = model_module.build_model(len(features), float(tr.get('learning_rate', .001)))
    history = model.fit(xtr, ytr, sample_weight=wt,
                        epochs=int(tr.get('epochs', 30)),
                        batch_size=int(tr.get('batch_size', 8192)),
                        shuffle=True, verbose=2)
    loss = list(map(float, history.history['loss']))
    bs = int(tr.get('inference_batch_size', 16384))
    scoretr, scorete = model_module.predict(model, xtr, bs), model_module.predict(model, xte, bs)

    saved = Path(general.get('saved_models_dir', 'data/saved_models')) / tr['model_name']
    saved.mkdir(parents=True, exist_ok=True)
    model.save(saved / 'model.keras')
    np.savez(saved / 'scaler.npz', mean=scaler.mean_, scale=scaler.scale_)
    meta = {'features': features, 'tree': general['tree'], 'weight_branch': general.get('weight_branch'),
            'backend': 'tensorflow', 'model_path': mc['model_path'], 'drop_negative_weight': tr.get('drop_negative_weight', False), 'absolute_negative_weight': tr.get('absolute_negative_weight', False)}
    (saved / 'metadata.json').write_text(json.dumps(meta, indent=2))
    (out / 'config_used.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))

    report = {'train': performance(ytr, scoretr, wtr),
              'test': performance(yte, scorete, wte),
              'parameters': int(model.count_params()),
              'epochs': len(loss), 'last_loss': loss[-1],
              'drop_negative_weight': meta['drop_negative_weight'], 'absolute_negative_weight': meta['absolute_negative_weight'],
              'weighted_metrics_note': 'Weighted ROC uses |MC weights|; yields use signed weights.',
              'processes': {}}
    for name in sorted(set(process)):
        report['processes'][str(name)] = {}
        for split, idx in [('train', itr), ('test', ite)]:
            sel = process[idx] == name
            ww = w[idx][sel]
            report['processes'][str(name)][split] = {
                'events': int(sel.sum()), 'signed_sumw': float(ww.sum()),
                'sumw2': float(np.square(ww).sum()),
                'negative_events': int((ww < 0).sum())}
    (out / 'report.json').write_text(json.dumps(report, indent=2))

    train, test = (ytr, scoretr, wtr), (yte, scorete, wte)
    if options.get('loss', True): plots.plot_loss(loss, out / 'loss.png')
    if options.get('correlation', True): plots.plot_correlation(xtr, features, out / 'correlation.png')
    if options.get('scores', True):
        for weighted in (False, True):
            plots.plot_scores(train, test, out / f'scores_{"weighted" if weighted else "unweighted"}.png', weighted=weighted)
    if options.get('roc', True):
        for weighted in (False, True):
            plots.plot_roc(train, test, out / f'roc_{"weighted" if weighted else "unweighted"}.png', weighted=weighted)
    if options.get('confusion', True):
        for weighted in (False, True):
            plots.plot_confusion(yte, scorete, wte, out / f'confusion_{"weighted" if weighted else "unweighted"}.png',
                                 weighted=weighted, threshold=float(tr.get('threshold', .5)))
    if options.get('significance', True):
        plots.plot_significance(yte, scorete, wte, out / 'significance.png',
                                float(tr.get('background_systematic_fraction', 0)))
    if options.get('stack', True):
        plots.plot_stack(yte, scorete, wte, process[ite], out / 'stack_yields.png')
        plots.plot_stack(yte, scorete, wte, process[ite], out / 'stack_normalized.png', normalize=True)
    if options.get('overlay', True):
        plots.plot_process_overlay(yte, scorete, wte, process[ite], out / 'process_overlay.png')
    print(f'\n[SAVED MODEL] {saved}\n[OUTPUT] {out}\n{json.dumps(report, indent=2)}')


if __name__ == '__main__':
    main()
