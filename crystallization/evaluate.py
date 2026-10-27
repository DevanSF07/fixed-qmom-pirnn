"""Inspect PIRNN predictions on noisy held-out histories. No model comparisons.
Window metrics use all test windows. Autonomous rollout is optional per batch;
long rollout accuracy is not guaranteed. Reports nonfinite outputs as failures.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .physics import set_orders,ORDERS,NAMES,INDEX,NREF,LREF,CREF
from .data import Windows
from .model import Forecaster

def load_model(checkpoint):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True);cfg=ck['config'];set_orders(cfg['moments'])
    model=Forecaster(cfg['training']['hidden']);model.load_state_dict(ck['model']);model.eval()
    return model,cfg

def predict(checkpoint,history,temperature):
    """Encoded noisy history [L,11], Celsius T [H+1] including current time.
    Returns scaled q [H+1,9] and z [H+1,10], including current estimate.
    """
    model,cfg=load_model(checkpoint);history=np.asarray(history);temperature=np.asarray(temperature)
    if history.shape!=(cfg['training']['history'],11) or temperature.ndim!=1 or len(temperature)<2:raise ValueError('Input shapes differ from checkpoint')
    with torch.no_grad():
        result=model(torch.tensor(history[None],dtype=torch.float32),torch.tensor(temperature[None],dtype=torch.float32))
    return {k:v[0].numpy() for k,v in result.items()}

def evaluate(data='outputs/data',checkpoint='outputs/pirnn/best.pt',out='outputs/evaluation',rollouts=1):
    torch.set_num_threads(2);model,cfg=load_model(checkpoint);s=cfg['training'];ds=Windows(data,'test',s['history'],s['horizon'])
    if cfg['physics']!=ds.cfg['physics'] or cfg['moments']!=ds.cfg['moments'] or not np.isclose(ds.dt,cfg['solver']['sample_interval_s']):raise ValueError('Checkpoint/data physics or timestep differs')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);squared=np.zeros(10);n=0
    with torch.no_grad():
        for h,T,y in DataLoader(ds,batch_size=s['batch_size']):
            pred=model(h,T)['z'][:,1:]
            if not torch.isfinite(pred).all():raise RuntimeError('Nonfinite window predictions')
            squared+=((pred-y[:,1:]).double().square().sum((0,1))).numpy();n+=pred.shape[0]*pred.shape[1]
    rmse=np.sqrt(squared/n);scales=np.r_[[NREF*LREF**(p+q) for p,q in ORDERS],CREF]
    manifest=json.loads((Path(data)/'manifest.json').read_text())
    metrics=dict(window_scaled_rmse=dict(zip(NAMES+['c'],rmse.tolist())),window_physical_rmse=dict(zip(NAMES+['c'],(rmse*scales).tolist())),test_batches=len(ds.batches),failed_generation_batches=len(manifest['failed']),autonomous=[],scope='Successful simulated cases only; no experimental validation')
    factor=ds.par.rho*ds.par.thickness*NREF*LREF**2/CREF;L=s['history']
    with torch.no_grad():
        for batch in ds.batches[:max(0,rollouts)]:
            obs=batch['observed'][:L];h=np.column_stack((np.log1p(obs[:,:10]),(obs[:,10]-30)/5))
            z=model(torch.tensor(h[None],dtype=torch.float32),torch.tensor(batch['T'][L-1:][None],dtype=torch.float32))['z'][0,1:].numpy();truth=batch['clean'][L:]
            if not np.isfinite(z).all():
                metrics['autonomous'].append(dict(id=batch['id'],status='NONFINITE_PREDICTION'));continue
            mass=z[:,9]+factor*z[:,INDEX[1,1]];m0=batch['clean'][L-1,9]+factor*batch['clean'][L-1,INDEX[1,1]]
            row=dict(id=batch['id'],status='finite',scaled_rmse=np.sqrt(np.mean((z-truth)**2,axis=0)).tolist(),max_mass_error_scaled=float(np.max(abs(mass-m0))))
            for name,idx in [('length_um',INDEX[1,0]),('width_um',INDEX[0,1])]:
                row[name+'_rmse']=float(np.sqrt(np.mean((z[:,idx]/z[:,0]-truth[:,idx]/truth[:,0])**2))*LREF*1e6)
            metrics['autonomous'].append(row)
            np.savez_compressed(out/(batch['id']+'.npz'),prediction=z,truth=truth,t=batch['t'][L:])
            fig,axes=plt.subplots(5,2,figsize=(12,15))
            for j,ax in enumerate(axes.flat):
                ax.plot(batch['t'][L:],truth[:,j],label='Clean simulator');ax.plot(batch['t'][L:],z[:,j],label='PIRNN');ax.set_title((NAMES+['c'])[j]+' scaled');ax.set_xlabel('seconds');ax.legend()
            fig.tight_layout();fig.savefig(out/(batch['id']+'.png'),dpi=130);plt.close(fig)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2));return metrics

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',default='outputs/data');p.add_argument('--checkpoint',default='outputs/pirnn/best.pt');p.add_argument('--out',default='outputs/evaluation');p.add_argument('--rollouts',type=int,default=1)
    print(json.dumps(evaluate(**vars(p.parse_args())),indent=2))
