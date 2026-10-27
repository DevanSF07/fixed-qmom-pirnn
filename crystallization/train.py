"""PIRNN only. No baseline/comparison experiment. No future noisy targets.
Validation error selects a checkpoint; clean test data are never used in fitting.
"""
import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .physics import load_config
from .data import Windows
from .model import Forecaster
from .losses import losses

def train(data='outputs/data',out='outputs/pirnn',config='config.json',epochs=None):
    cfg,par=load_config(config);s=cfg['training'];epochs=epochs if epochs is not None else s['epochs']
    if epochs<1:raise ValueError('epochs must be positive')
    seed=cfg['random_seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4)
    tr=Windows(data,'train',s['history'],s['horizon']);va=Windows(data,'val',s['history'],s['horizon'])
    if cfg['physics']!=tr.cfg['physics'] or cfg['moments']!=tr.cfg['moments']:raise ValueError('Dataset physics differs from training configuration')
    if cfg['noise']!=tr.cfg['noise']:raise ValueError('Noise configuration differs from saved observations')
    if not np.isclose(tr.dt,cfg['solver']['sample_interval_s']):raise ValueError('Sampling interval mismatch')
    if tr.cfg['physics']!=va.cfg['physics']:raise ValueError('Validation parameters differ')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    model=Forecaster(s['hidden'],par=par);opt=torch.optim.Adam(model.parameters(),lr=s['learning_rate'])
    logs=[];best=float('inf');failures=len(json.loads((Path(data)/'manifest.json').read_text())['failed'])
    print(f'Train windows={len(tr)}, val windows={len(va)}, failed batches={failures}. Metrics cover successful cases only.')
    for epoch in range(epochs):
        model.train();sums=dict(data=0.,physics=0.,mass=0.);count=0
        for hist,T,y in DataLoader(tr,batch_size=s['batch_size'],shuffle=True):
            opt.zero_grad();terms=losses(model(hist,T),y,T,tr.dt,par)
            loss=terms['data']+s['physics_weight']*terms['physics']+s['mass_weight']*terms['mass']
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),s['gradient_clip']);opt.step()
            for key in sums:sums[key]+=terms[key].item()*len(hist)
            count+=len(hist)
        model.eval();value=0.;n=0
        with torch.no_grad():
            for hist,T,y in DataLoader(va,batch_size=s['batch_size']):
                value+=losses(model(hist,T),y,T,va.dt,par)['data'].item()*len(hist);n+=len(hist)
        row=dict(epoch=epoch+1,**{k:v/count for k,v in sums.items()},val_log_mse=value/n);logs.append(row)
        print(json.dumps(row),flush=True)
        if value/n<best:
            best=value/n;torch.save(dict(model=model.state_dict(),config=cfg,epochs_requested=epochs,data_manifest=str(Path(data)/'manifest.json')),out/'best.pt')
        (out/'training.json').write_text(json.dumps(logs,indent=2))
    return logs

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='config.json');p.add_argument('--data',default='outputs/data');p.add_argument('--out',default='outputs/pirnn');p.add_argument('--epochs',type=int)
    train(**vars(p.parse_args()))
