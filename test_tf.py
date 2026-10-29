#!/usr/bin/env python3
"""Evaluate a saved TensorFlow model on independent ROOT samples."""
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

from tools.data import read_samples
from tools.preprocess import apply_negative_weight_policy
from tools.models.loader import load_model_module
from tools.metrics import performance
from tools import plotting as plots


def main():
    parser = argparse.ArgumentParser(description='Test saved TensorFlow classifier')
    parser.add_argument('-c', '--config', default='test_config_tf.yaml')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_cfg=cfg['model']; folder=Path(model_cfg['model_path']).expanduser().parent
    metadata = json.loads((folder / 'metadata.json').read_text())
    scaler = np.load(model_cfg['scaler_path'])
    model = tf.keras.models.load_model(model_cfg['model_path'])
    general = cfg['general']
    x, y, w, process = read_samples(cfg['samples'], metadata['features'],
                                     metadata.get('weight_branch'),
                                     general.get('tree', metadata['tree']))
    x, y, w, process = apply_negative_weight_policy(x,y,w,process,**{k:general.get(k,False) for k in ('drop_negative_weight','absolute_negative_weight')})
    x = ((x - scaler['mean']) / scaler['scale']).astype('float32')
    module=load_model_module(metadata.get('model_path','tools/models/mlp_tensorflow.py'))
    scores = module.predict(model, x, int(general.get('inference_batch_size', 16384)))
    out = Path(general.get('output_dir', 'output/external_test_tf'))
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'predictions.npz', score=scores, label=y,
                        weight_signed=w, process=process.astype(str))
    options = cfg.get('plots', {})
    if options.get('stack', True):
        plots.plot_stack(y, scores, w, process, out / 'stack_yields.png')
        plots.plot_stack(y, scores, w, process, out / 'stack_normalized.png', normalize=True)
    if options.get('overlay', True):
        plots.plot_process_overlay(y, scores, w, process, out / 'process_overlay.png')
    if set(np.unique(y)) == {0, 1}:
        if options.get('confusion', True):
            plots.plot_confusion(y, scores, w, out / 'confusion.png', threshold=float(general.get('threshold', .5)))
        if options.get('significance', True):
            plots.plot_significance(y, scores, w, out / 'significance.png',
                                    float(general.get('background_systematic_fraction', 0)))
        if options.get('roc', True):
            fig, ax = plt.subplots(figsize=(6, 5))
            for weighted in (False, True):
                sw = np.abs(w) if weighted else None
                fpr, tpr, _ = roc_curve(y, scores, sample_weight=sw)
                auc = roc_auc_score(y, scores, sample_weight=sw)
                ax.plot(fpr, tpr, label=f'{"|w|" if weighted else "unweighted"}: AUC {auc:.4f}')
            ax.set(xlabel='Background efficiency', ylabel='Signal efficiency', title='External test ROC')
            ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
            fig.savefig(out / 'roc.png'); plt.close(fig)
        report = performance(y, scores, w, float(general.get('threshold', .5)))
    else:
        report = {'events': int(len(y)), 'signed_sumw': float(w.sum()),
                  'note': 'ROC/confusion require both classes'}
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    print(f'[OUTPUT] {out}\n{json.dumps(report, indent=2)}')


if __name__ == '__main__':
    main()
