#!/usr/bin/env python3
"""Train the TensorFlow backend; original PyTorch train.py stays unchanged."""
import argparse
import yaml
from pathlib import Path

import uproot
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch

from tools.data import read_samples
from tools.preprocess import apply_negative_weight_policy, split_scale, balance_training_weights

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

def plot_variable(features,X_train,X_test,y_train,y_test,w_train,w_test,scaler,branch, bins,*,outputdir,logy=True):
    k = features.index(branch)
    centers = (bins[:-1] + bins[1:]) / 2
    colors = {0: 'tab:blue', 1: 'tab:red'}

    # Convert standardized features back to physical values
    train = X_train[:, k] * scaler.scale_[k] + scaler.mean_[k]
    test = X_test[:, k] * scaler.scale_[k] + scaler.mean_[k]

    #plot figure            
    fig, ax = plt.subplots(figsize=(7, 5))
    for cls, name in [(0, 'Background'), (1, 'Signal')]:
        tr, te = y_train == cls, y_test == cls

        h = np.histogram(train[tr], bins, weights=w_train[tr])[0]
        d = np.histogram(test[te], bins, weights=w_test[te])[0]
        e = np.sqrt(np.histogram(test[te], bins, weights=w_test[te]**2)[0])

        if h.sum() > 0: h = h / h.sum()
        if d.sum() > 0:
            e = e / d.sum()
            d = d / d.sum()

        ax.stairs(np.where(h > 0, h, np.nan), bins, fill=True,
                  alpha=0.25, color=colors[cls], label=f'{name} train')

        mask = d > 0
        ax.errorbar(centers[mask], d[mask], yerr=e[mask],
                    fmt='o', ms=3.5, capsize=2,
                    color=colors[cls], label=f'{name} test')

    ax.set(xlabel=branch, ylabel='Fraction of events / bin')
    if logy:
        ax.set_yscale('log')
        ax.set_ylim(1e-5, 1.0)

    ax.grid(alpha=0.2, which='both')
    ax.legend()
    plt.tight_layout()
    plt.savefig(outputdir / f'{branch}.png',dpi=250)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Train HEP classifier with TensorFlow')
    parser.add_argument('-c', '--config', default='config_tf.yaml')
    parser.add_argument('-e', '--max_events',type=int,default=None,help='Maximum number of events to read per process (default: all)')
    args = parser.parse_args()

    # custom options
    balance_class_weight=False

    # read yaml config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    general, tr, mc, options = cfg['general'], cfg['training'], cfg['model'], cfg.get('plots', {})

    # random Seed
    seed = int(tr.get('seed', 42))
    np.random.seed(seed)

    out = Path(general.get('output_dir', 'output')) / "input_distribution"
    out.mkdir(parents=True, exist_ok=True)
    features = general['features']
    # Always load signed MC weights for reporting, regardless of training switches.
    x, y, w, process = read_samples(cfg['samples'], features, general.get('weight_branch'), general['tree'],max_events = args.max_events)

    # Remove negative weights OR make absolute weights
    x, y, w, process = apply_negative_weight_policy(x, y, w, process, **{k: tr.get(k, False) for k in ('drop_negative_weight', 'absolute_negative_weight')})
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Training requires both labels 0 and 1')   

    # Split into train and test dataset
    itr, ite, scaler, X_train, X_test = split_scale(x,y,w,process,test_fraction=float(tr.get('test_fraction',.2)),seed=seed)
    y_train, y_test = y[itr], y[ite]
    w_train, w_test = w[itr], w[ite]
    
    ##Weight study
    true_class_weights = [w[y == 0].sum(), w[y == 1].sum()]
    print("true class weights: ",true_class_weights)
    if balance_class_weight:
        class_weights_train = (w_train[y_train == 0].sum(), w_train[y_train == 1].sum())
        for i in range(len(class_weights_train)):
            w_train[y_train == i] *= (max(class_weights_train) / class_weights_train[i])  # equalize number of background and signal event
            w_test[y_test == i] *= (true_class_weights[i] / w_test[y_test == i].sum())    # increase test weight to compensate for sampling

        print("class_weights_train bkg :", class_weights_train[0])
        print("class_weights_train sig :", class_weights_train[1])
        print("After balancing class weights\n=========================")
        print("total weight sig (train):", w_train[y_train == 1].sum())
        print("total weight bkg (train):", w_train[y_train == 0].sum())
        print("total weight sig (test) :", w_test[y_test == 1].sum())
        print("total weight bkg (test) :", w_test[y_test == 0].sum())

    #print stat
    print(f"{'Split':<8} {'Class':<12} {'Events':>12} {'Weighted yield':>16} {'MC stat.':>12}")
    print("-" * 64)
    for split, labels, weights in [
            ("Train", y_train, w_train),
            ("Test",  y_test,  w_test),
    ]:
        labels = np.asarray(labels).ravel()
        weights = np.asarray(weights, dtype=np.float64).ravel()
        
        for cls, name in [(0, "Background"), (1, "Signal")]:
            mask = labels == cls
            selected = weights[mask]

            n_events = np.count_nonzero(mask)
            weighted_yield = selected.sum()
            stat_error = np.sqrt(np.square(selected).sum())

            print(
                f"{split:<8} {name:<12} {n_events:>12,d} "
                f"{weighted_yield:>16,.3f} {stat_error:>12,.3f}"
            )            
        
    ##Plot input features
    plot_variable(features,X_train,X_test,y_train,y_test,w_train,w_test,scaler,"lep_pt_lead_NOSYS", np.linspace(0, 300, 61),outputdir=out)
    plot_variable(features,X_train,X_test,y_train,y_test,w_train,w_test,scaler,"lep_pt_sublead_NOSYS", np.linspace(0, 200, 41),outputdir=out)
    plot_variable(features,X_train,X_test,y_train,y_test,w_train,w_test,scaler,"lep_pt_total_NOSYS", np.linspace(0, 600, 121),outputdir=out)

    # Jet multiplicities
    plot_variable(features,X_train,X_test,y_train,y_test,w_train,w_test,scaler,"njets_NOSYS", np.linspace(-0.5,10.5,11),outputdir=out)
    plot_variable(features,X_train,X_test,y_train,y_test,w_train,w_test,scaler,"bjets_num_NOSYS", np.linspace(-0.5,10.5,11),outputdir=out)

    # Mass
    plot_variable(features,X_train,X_test,y_train,y_test,w_train,w_test,scaler,"mll_NOSYS", np.linspace(65,115,61),outputdir=out)
    plot_variable(features,X_train,X_test,y_train,y_test,w_train,w_test,scaler,"mt_ll_NOSYS", np.linspace(0,500,101),outputdir=out)

    # Angular variables
    plot_variable(features, X_train, X_test, y_train, y_test, w_train, w_test, scaler,"deltaR_ll_NOSYS", np.linspace(0, 6, 61), outputdir=out)
    plot_variable(features, X_train, X_test, y_train, y_test, w_train, w_test, scaler,"dphi_llmet_NOSYS", np.linspace(0, np.pi, 51), outputdir=out)
    plot_variable(features, X_train, X_test, y_train, y_test, w_train, w_test, scaler,"dphi_leadjet_met_NOSYS", np.linspace(0, np.pi, 51), outputdir=out)
    
    # Missing transverse momentum (GeV)
    plot_variable(features, X_train, X_test, y_train, y_test, w_train, w_test, scaler,"met_met_NOSYS", np.linspace(0, 800, 101), outputdir=out)
    
    # MET significance (dimensionless)
    plot_variable(features, X_train, X_test, y_train, y_test, w_train, w_test, scaler,"met_significance_NOSYS", np.linspace(0, 50, 101), outputdir=out)
    
    # Dimensionless ratios
    plot_variable(features, X_train, X_test, y_train, y_test, w_train, w_test, scaler,"met_over_mtz_NOSYS", np.linspace(0, 10, 101), outputdir=out)
    plot_variable(features, X_train, X_test, y_train, y_test, w_train, w_test, scaler,"pt_over_mtz_NOSYS", np.linspace(0, 10, 101), outputdir=out)
    
if __name__ == '__main__':
    main()
