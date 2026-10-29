from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from .metrics import scan_significance


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loss(history, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(history)+1), history)
    ax.set(xlabel='Epoch', ylabel='Weighted BCE', title='Training loss')
    ax.grid(alpha=.25)
    _save(fig, path)


def plot_correlation(x, features, path, max_events=150000, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=min(len(x), max_events), replace=False)
    corr = np.corrcoef(x[idx], rowvar=False)
    fig, ax = plt.subplots(figsize=(max(8, len(features)*.6), max(7, len(features)*.6)))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
    ax.set(xticks=range(len(features)), yticks=range(len(features)), title='Training feature correlation (unweighted)')
    ax.set_xticklabels(features, rotation=90, fontsize=8)
    ax.set_yticklabels(features, fontsize=8)
    fig.colorbar(im, ax=ax, label='Pearson r')
    _save(fig, path)


def plot_confusion(y, score, weights, path, weighted=False, threshold=.5):
    matrix = confusion_matrix(y, score >= threshold, labels=[0,1], sample_weight=np.abs(weights) if weighted else None)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(matrix, cmap='Blues')
    ax.set(xticks=[0,1], yticks=[0,1], xlabel='Predicted', ylabel='True', title='Confusion matrix' + (' (|weight|)' if weighted else ''))
    for i in range(2):
        for j in range(2): ax.text(j, i, f'{matrix[i,j]:.4g}', ha='center', va='center')
    _save(fig, path)


def plot_scores(train, test, path, weighted=False, bins=40):
    edges = np.linspace(0,1,bins+1); centers=(edges[1:]+edges[:-1])/2
    fig, ax = plt.subplots(figsize=(7,5))
    for c, name, color in [(0,'Background','tab:blue'),(1,'Signal','tab:red')]:
        for data, is_train in [(train,True),(test,False)]:
            y, score, w = data
            sel=y==c; ww=np.abs(w[sel]) if weighted else np.ones(sel.sum())
            h=np.histogram(score[sel],edges,weights=ww)[0]
            e=np.sqrt(np.histogram(score[sel],edges,weights=ww**2)[0]); norm=h.sum()
            if norm <= 0: continue
            h=h/norm; e=e/norm
            if is_train:
                ax.stairs(h, edges, color=color, fill=True, alpha=.23, label=f'{name} train')
            else:
                use=h>0
                ax.errorbar(centers[use],h[use],yerr=e[use],color=color,fmt='o',ms=3,capsize=2,label=f'{name} test')
    ax.set(xlabel='NN score', ylabel='Fraction / bin', title='Score shape' + (' (|MC weights|)' if weighted else ' (unweighted)'))
    ax.set_yscale('log'); ax.legend(); ax.grid(alpha=.2)
    _save(fig,path)


def plot_roc(train, test, path, weighted=False):
    fig, ax=plt.subplots(figsize=(6,5))
    for name, data, color in [('Train',train,'tab:blue'),('Test',test,'tab:red')]:
        y, score, w=data; ww=np.abs(w) if weighted else None
        fpr,tpr,_=roc_curve(y,score,sample_weight=ww)
        auc=roc_auc_score(y,score,sample_weight=ww)
        ax.plot(fpr,tpr,color=color,label=f'{name}: AUC {auc:.4f}')
    ax.plot([0,1],[0,1],'k--',alpha=.4)
    ax.set(xlabel='Background efficiency',ylabel='Signal efficiency',title='ROC' + (' (|MC weights|)' if weighted else ' (unweighted)'))
    ax.legend(); ax.grid(alpha=.2); _save(fig,path)


def plot_significance(y, score, weights, path, frac_bkg_systematic=0.):
    cuts,s,b,z=scan_significance(y,score,weights,frac_bkg_systematic=frac_bkg_systematic)
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot(cuts,z)
    ax.set(xlabel='NN threshold',ylabel='S / sqrt(B + (fB)^2)',title=f'Expected weighted yields; f={frac_bkg_systematic:g}')
    ax.grid(alpha=.2);_save(fig,path)


def plot_stack(y, score, weights, processes, path, bins=40, normalize=False):
    """Stack signed-weight background bins; overlay signal line. Signed bins may be negative."""
    edges=np.linspace(0,1,bins+1)
    bkg_names=sorted(set(processes[y==0])); sig_names=sorted(set(processes[y==1]))
    bkg=[np.histogram(score[(processes==p)&(y==0)],edges,weights=weights[(processes==p)&(y==0)])[0] for p in bkg_names]
    sig=sum((np.histogram(score[(processes==p)&(y==1)],edges,weights=weights[(processes==p)&(y==1)])[0] for p in sig_names),np.zeros(bins))
    if normalize:
        denom=sum(np.sum(h) for h in bkg)
        if denom > 0: bkg=[h/denom for h in bkg]
        if sig.sum()>0: sig=sig/sig.sum()
    fig,ax=plt.subplots(figsize=(8,5)); centers=(edges[1:]+edges[:-1])/2
    bottom=np.zeros(bins)
    for p,h in zip(bkg_names,bkg):
        ax.bar(centers,h,bottom=bottom,width=np.diff(edges),align='center',label=p,alpha=.7)
        bottom+=h
    ax.stairs(sig,edges,color='red',lw=2,label='Signal')
    ax.set(xlabel='NN score',ylabel='Normalized shape' if normalize else 'Weighted yield',title='Background stack + signal')
    ax.semilogy()
    ax.legend(fontsize=8)
    _save(fig,path)


def plot_process_overlay(y, score, weights, processes, path, bins=40, weighted=True):
    edges=np.linspace(0,1,bins+1)
    fig,ax=plt.subplots(figsize=(8,5))
    for p in sorted(set(processes)):
        mask=processes==p; w=np.abs(weights[mask]) if weighted else None
        h=np.histogram(score[mask],edges,weights=w)[0]; total=h.sum()
        if total>0: ax.stairs(h/total,edges,label=p,lw=1.5)
    ax.set(xlabel='NN score',ylabel='Fraction / bin',title='Unit-area process score overlay')
    ax.semilogy()
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    _save(fig,path)
