#!/usr/bin/env python3
import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tools.data import read_samples
from tools.preprocess import apply_negative_weight_policy, split_scale, balance_training_weights
from tools.models.loader import load_model_module
from tools.metrics import performance
from tools import plotting as plots


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('-c','--config',default='config.yaml')
    args=parser.parse_args()
    with open(args.config) as f: cfg=yaml.safe_load(f)
    general=cfg['general']; tr=cfg['training']; mc=cfg['model']; options=cfg.get('plots',{})
    seed=int(tr.get('seed',42)); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(int(tr.get('cpu_threads',4)))
    out=Path(general.get('output_dir','output')) / datetime.now().strftime('training_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    features=general['features']; weight_branch=general.get('weight_branch') if tr.get('use_weights',True) else None
    # Read the MC branch even if training is unweighted: physics yields and plots need it.
    x,y,w,process=read_samples(cfg['samples'],features,general.get('weight_branch'),general['tree'])
    x,y,w,process=apply_negative_weight_policy(x,y,w,process, **{k:tr.get(k,False) for k in ('drop_negative_weight','absolute_negative_weight')})
    if len(np.unique(y))!=2: raise ValueError('Need both signal (1) and background (0)')
    itr,ite,scaler,xtr,xte=split_scale(x,y,w,process,test_fraction=float(tr.get('test_fraction',.2)),seed=seed)
    ytr=y[itr]; yte=y[ite]; wtr=w[itr]; wte=w[ite]
    wt=balance_training_weights(wtr,ytr,tr.get('use_weights',True),tr.get('balanced_sample_weight',True))
    loader=DataLoader(TensorDataset(torch.from_numpy(xtr),torch.from_numpy(ytr.astype('float32')).unsqueeze(1),torch.from_numpy(wt).unsqueeze(1)),batch_size=int(tr.get('batch_size',512)),shuffle=True)
    model_module=load_model_module(mc['model_path']); model=model_module.build_model(len(features)); opt=torch.optim.Adam(model.parameters(),lr=float(tr.get('learning_rate',.001)))
    criterion=nn.BCEWithLogitsLoss(reduction='none'); hist=[]
    for epoch in range(int(tr.get('epochs',30))):
        model.train(); total=0
        for xb,yb,wb in loader:
            opt.zero_grad(); loss=(criterion(model(xb),yb)*wb).mean();loss.backward();opt.step()
            total+=loss.item()*len(xb)
        value=total/len(xtr);hist.append(value)
        print(f'Epoch {epoch+1:03d} loss={value:.6f}')
    bs=int(tr.get('inference_batch_size',16384)); str_=model_module.predict(model,xtr,bs); ste=model_module.predict(model,xte,bs)
    checkpoint={'state_dict':model.state_dict(),'features':features,'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'tree':general['tree'],'weight_branch':general.get('weight_branch'), 'samples_config':cfg['samples']}
    saved=Path(general.get('saved_models_dir','data/saved_models'))/tr['model_name']; saved.mkdir(parents=True,exist_ok=True)
    checkpoint['model_path']=mc['model_path']
    torch.save(checkpoint,saved/'model.pt')
    np.savez(saved/'scaler.npz',mean=scaler.mean_,scale=scaler.scale_)
    (saved/'metadata.json').write_text(json.dumps({'features':features,'model_path':mc['model_path'],'tree':general['tree'],'weight_branch':general.get('weight_branch')},indent=2))
    # JSON makes the scaler portable without pickle.
    (out/'config_used.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    report={'train':performance(ytr,str_,wtr),'test':performance(yte,ste,wte),'parameters':sum(p.numel() for p in model.parameters()),'epochs':len(hist),'last_loss':hist[-1], 'drop_negative_weight':tr.get('drop_negative_weight',False),'absolute_negative_weight':tr.get('absolute_negative_weight',False), 'weighted_metrics_note':'Weighted ROC uses absolute MC weights; yields use original signed weights.'}
    report['processes'] = {}
    for name in sorted(set(process)):
        report['processes'][str(name)] = {}
        for split, idx in [('train', itr), ('test', ite)]:
            m = process[idx] == name
            weights = w[idx][m]
            report['processes'][str(name)][split] = {
                'events': int(m.sum()), 'signed_sumw': float(weights.sum()),
                'sumw2': float(np.square(weights).sum()),
                'negative_events': int((weights < 0).sum())
            }
    (out/'report.json').write_text(json.dumps(report,indent=2))
    train=(ytr,str_,wtr);test=(yte,ste,wte)
    if options.get('loss',True): plots.plot_loss(hist,out/'loss.png')
    if options.get('correlation',True): plots.plot_correlation(xtr,features,out/'correlation.png')
    if options.get('scores',True):
        for weighted in (False,True): plots.plot_scores(train,test,out/f'scores_{"weighted" if weighted else "unweighted"}.png',weighted=weighted)
    if options.get('roc',True):
        for weighted in (False,True): plots.plot_roc(train,test,out/f'roc_{"weighted" if weighted else "unweighted"}.png',weighted=weighted)
    if options.get('confusion',True):
        for weighted in (False,True):plots.plot_confusion(yte,ste,wte,out/f'confusion_{"weighted" if weighted else "unweighted"}.png',weighted=weighted,threshold=tr.get('threshold',.5))
    if options.get('significance',True): plots.plot_significance(yte,ste,wte,out/'significance.png',float(tr.get('background_systematic_fraction',0)))
    if options.get('stack',True):
        plots.plot_stack(yte,ste,wte,process[ite],out/'stack_yields.png')
        plots.plot_stack(yte,ste,wte,process[ite],out/'stack_normalized.png',normalize=True)
    if options.get('overlay',True):plots.plot_process_overlay(yte,ste,wte,process[ite],out/'process_overlay.png')
    print(f'\n[SAVED MODEL] {saved}\n[OUTPUT] {out}\n{json.dumps(report,indent=2)}')

if __name__=='__main__':main()
