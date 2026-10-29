#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from tools.data import read_samples
from tools.preprocess import apply_negative_weight_policy
from tools.models.loader import load_model_module
from tools.metrics import performance
from tools import plotting as plots


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('-c','--config',default='test_config.yaml')
    args=parser.parse_args()
    with open(args.config) as f:cfg=yaml.safe_load(f)
    model_cfg=cfg['model']; root=Path(model_cfg['model_path']).expanduser()
    # Only load checkpoints you created/trust.
    cp=torch.load(root,map_location='cpu',weights_only=True)
    features=cp['features']; general=cfg['general']
    x,y,w,process=read_samples(cfg['samples'],features,cp.get('weight_branch'),general.get('tree',cp['tree']))
    x,y,w,process=apply_negative_weight_policy(x,y,w,process,**{k:general.get(k,False) for k in ('drop_negative_weight','absolute_negative_weight')})
    scaler=np.load(model_cfg['scaler_path']); x=((x-scaler['mean'])/scaler['scale']).astype('float32')
    module=load_model_module(cp.get('model_path',model_cfg.get('model_source','tools/models/mlp_torch.py')))
    model=module.build_model(len(features));model.load_state_dict(cp['state_dict'])
    score=module.predict(model,x,general.get('inference_batch_size',16384))
    output=Path(general.get('output_dir','output/test'));output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'predictions.npz',score=score,label=y,weight_signed=w,process=process.astype(str))
    p=cfg.get('plots',{})
    if p.get('stack',True):
        plots.plot_stack(y,score,w,process,output/'stack_yields.png')
        plots.plot_stack(y,score,w,process,output/'stack_normalized.png',normalize=True)
    if p.get('overlay',True):plots.plot_process_overlay(y,score,w,process,output/'process_overlay.png')
    if len(np.unique(y))==2:
        if p.get('confusion',True):
            plots.plot_confusion(y,score,w,output/'confusion.png',threshold=general.get('threshold',.5))
        if p.get('significance',True):plots.plot_significance(y,score,w,output/'significance.png',float(general.get('background_systematic_fraction',0)))
        if p.get('roc',True):
            # An external testing sample has no training curve; plot its ROC separately.
            from sklearn.metrics import roc_curve, roc_auc_score
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots(figsize=(6,5))
            for weighted in (False,True):
                sw=np.abs(w) if weighted else None
                fpr,tpr,_=roc_curve(y,score,sample_weight=sw)
                auc=roc_auc_score(y,score,sample_weight=sw)
                ax.plot(fpr,tpr,label=f'{"|w|" if weighted else "unweighted"}: AUC {auc:.4f}')
            ax.set(xlabel='Background efficiency',ylabel='Signal efficiency',title='External test ROC')
            ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output/'roc.png');plt.close(fig)
        results=performance(y,score,w,general.get('threshold',.5))
    else:
        results={'events':int(len(y)),'signed_sumw':float(w.sum()),'note':'ROC/confusion require both labels'}
    (output/'report.json').write_text(json.dumps(results,indent=2))
    print(f'[OUTPUT] {output}\n{json.dumps(results,indent=2)}')

if __name__=='__main__':main()
