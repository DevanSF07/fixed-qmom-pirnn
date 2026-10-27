"""Plan experiments, generate complete batches, and create noisy observations.
Three-node seed shapes are sampled directly, not claimed exact lognormal fits.
Seed mass fraction is relative to initial dissolved solute mass.
Noise is a synthetic observation assumption, not a calibrated instrument model.
"""
import argparse,json
from pathlib import Path
import numpy as np
from .physics import *
from .profiles import Profile,FAMILIES

def make_design(cfg):
    d=cfg['design'];par=Parameters(**cfg['physics']);rng=np.random.default_rng(cfg['random_seed'])
    groups=d['seed_groups'];dt=cfg['solver']['sample_interval_s']
    if groups<10 or d['settings_per_group']<1 or dt<=0:raise ValueError('Require >=10 groups, positive settings and dt')
    if abs(sum(cfg['split'].values())-1)>1e-9:raise ValueError('Split fractions must sum to 1')
    for key in ('seeding_temperature_C','hot_offset_C','mean_cooling_rate_factor','seed_mass_fraction','node_length_factors','node_width_factors','relative_weight_factors'):
        v=d[key]
        if len(v)!=2 or v[0]<=0 or v[1]<v[0]:raise ValueError('Invalid positive range: '+key)
    if d['seeding_temperature_C'][0]<=d['final_temperature_C'] or d['final_temperature_C']<=0:raise ValueError('Require Th > Ts > Tf > 0')
    permutation=rng.permutation(groups);a=int(cfg['split']['train']*groups);b=int((cfg['split']['train']+cfg['split']['val'])*groups)
    split={int(g):('train' if j<a else 'val' if j<b else 'test') for j,g in enumerate(permutation)}
    cases=[]
    for group in range(groups):
        ref=reference_state()[:9]
        scale_w = rng.uniform(*d['relative_weight_factors'], 3)
        scale_L = rng.uniform(*d['node_length_factors'], 3)
        scale_W = rng.uniform(*d['node_width_factors'], 3)
        lengths = np.sort(ref[3:6] * scale_L)
        if lengths[1] - lengths[0] < 0.02: lengths[1] = lengths[0] + 0.02
        if lengths[2] - lengths[1] < 0.03: lengths[2] = lengths[1] + 0.03
        widths = ref[6:9] * scale_W
        q = np.r_[ref[:3] * scale_w, lengths, widths]
        mass_fraction=float(rng.uniform(*d['seed_mass_fraction']))
        for setting in range(d['settings_per_group']):
            Ts=float(rng.uniform(*d['seeding_temperature_C']));Th=Ts+float(rng.uniform(*d['hot_offset_C']))
            Tf=float(d['final_temperature_C']);c0=float(solubility(Th,par))
            rate=float(d['reference_cooling_rate_C_s']*rng.uniform(*d['mean_cooling_rate_factor']))
            duration=float(np.ceil((Ts-Tf)/rate/dt)*dt)
            population=q.copy();raw_mass=par.rho*par.thickness*NREF*LREF**2*moments(q)[INDEX[1,1]]
            population[:3]*=mass_fraction*c0/raw_mass
            initial=np.r_[population,c0/CREF,Ts]
            for family in cfg['profiles']:
                if family not in FAMILIES:raise ValueError('Unknown profile: '+family)
                profile=Profile(family,duration,Ts,Tf,rng)
                cases.append(dict(id=f'g{group:03d}_s{setting:02d}_{family}',index=len(cases),group=group,setting=setting,split=split[group],Th_C=Th,Ts_C=Ts,Tf_C=Tf,c0_kg_m3=c0,sigma0=c0/float(solubility(Ts,par))-1,seed_mass_fraction=mass_fraction,average_cooling_rate_C_s=(Ts-Tf)/duration,initial_state11_scaled=initial.tolist(),profile=profile.metadata()))
    return cases

def add_observations(path,noise,seed):
    with np.load(path,allow_pickle=False) as f:payload={k:f[k] for k in f.files}
    rng=np.random.default_rng(seed);q=payload['q'].copy()
    def factors(cv,shape):
        if cv<0:raise ValueError('Noise CV must be nonnegative')
        s=np.sqrt(np.log1p(cv*cv));return np.exp(s*rng.normal(size=shape)-s*s/2)
    for sl,key in ((slice(0,3),'node_weight_cv'),(slice(3,6),'node_length_cv'),(slice(6,9),'node_width_cv')):
        q[:,sl]*=factors(noise[key],q[:,sl].shape)
    obs_m=moments(q)
    c=payload['clean'][:,9]+rng.normal(0,noise['concentration_std_kg_m3']/CREF,len(q))
    T=payload['T']+rng.normal(0,noise['temperature_std_C'],len(q))
    payload['observed']=np.column_stack((obs_m,c,T));payload['noise']=json.dumps(noise)
    payload['noise_seed']=seed
    np.savez_compressed(path,**payload)

def generate(cfg,out='outputs/data',limit=None):
    from .simulate import run_case
    out=Path(out)
    if (out/'manifest.json').exists():raise FileExistsError('Choose a new dataset folder; existing manifest retained')
    out.mkdir(parents=True,exist_ok=True);cases=make_design(cfg)
    if limit is not None:
        if limit<1:raise ValueError('limit must be positive')
        cases=cases[:limit]
    manifest=dict(config=cfg,planned=len(cases),completed=[],failed=[])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for case in cases:
        path=out/(case['id']+'.npz')
        try:
            if path.exists():raise FileExistsError(str(path))
            run_case(case,cfg,path)
            noise_seed=int(np.random.SeedSequence([cfg['random_seed'],case['index'],471]).generate_state(1)[0])
            add_observations(path,cfg['noise'],noise_seed)
            manifest['completed'].append(dict(file=path.name,group=case['group'],split=case['split'],id=case['id']))
            print('SAVED',case['id'],flush=True)
        except QuadratureFailure as e:
            manifest['failed'].append(dict(id=case['id'],group=case['group'],split=case['split'],case=case,failure=e.details))
            print('FAILED',case['id'],str(e),flush=True)
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(planned=len(cases),completed=len(manifest['completed']),failed=len(manifest['failed'])),indent=2))
    return manifest

class Windows:
    def __init__(self,folder,split,history=5,horizon=10):
        import torch
        folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text())
        self.cfg=manifest['config'];set_orders(self.cfg['moments']);self.par=Parameters(**self.cfg['physics'])
        self.dt=float(self.cfg['solver']['sample_interval_s']);self.history=history;self.horizon=horizon
        if min(history,horizon)<1:raise ValueError('Positive history and horizon required')
        group_splits={}
        for entry in manifest['completed']:
            g=entry['group'];s=entry['split']
            if g in group_splits and group_splits[g]!=s:raise ValueError('Seed-group split leakage')
            group_splits[g]=s
        self.batches=[];self.ids=[]
        for entry in manifest['completed']:
            if entry['split']!=split:continue
            with np.load(folder/entry['file'],allow_pickle=False) as f:
                if not np.array_equal(f['orders'],ORDERS):raise ValueError('Moment-order mismatch')
                if json.loads(str(f['parameters']))!=self.par.to_dict():raise ValueError('Parameter mismatch')
                if not np.allclose(np.diff(f['t']),self.dt):raise ValueError('Observation interval mismatch')
                batch=dict(clean=f['clean'].astype('float32'),observed=f['observed'].astype('float32'),T=f['T'].astype('float32'),t=f['t'],id=entry['id'])
            if not np.isfinite(batch['observed']).all() or np.any(batch['observed'][:,:10]<=-1):raise ValueError('Observations cannot be log1p encoded')
            b=len(self.batches);self.batches.append(batch)
            self.ids.extend((b,k) for k in range(history-1,len(batch['T'])-horizon))
        if not self.ids:raise ValueError('No usable windows for '+split+'; inspect successful/failed batches and lengths')
    def __len__(self):return len(self.ids)
    def __getitem__(self,i):
        import torch
        b,k=self.ids[i];v=self.batches[b];L=self.history;H=self.horizon
        obs=v['observed'][k-L+1:k+1]
        history=np.column_stack((np.log1p(obs[:,:10]),(obs[:,10]-30)/5)).astype('float32')
        return tuple(torch.tensor(a.copy()) for a in (history,v['T'][k:k+H+1],v['clean'][k:k+H+1]))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    plan=sub.add_parser('plan');plan.add_argument('--config',default='config.json');plan.add_argument('--out',default='outputs/design.json')
    gen=sub.add_parser('generate');gen.add_argument('--config',default='config.json');gen.add_argument('--out',default='outputs/data');gen.add_argument('--limit',type=int)
    ins=sub.add_parser('inspect');ins.add_argument('file')
    a=p.parse_args()
    if a.command=='inspect':
        with np.load(a.file,allow_pickle=False) as f:
            print(json.dumps(dict(arrays={k:list(f[k].shape) for k in f.files},case=json.loads(str(f['case'])),first_clean=f['clean'][0].tolist(),first_observed=f['observed'][0].tolist() if 'observed' in f else None),indent=2))
    else:
        cfg,_=load_config(a.config)
        if a.command=='plan':
            cases=make_design(cfg);dest=Path(a.out);dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_text(json.dumps(dict(config=cfg,cases=cases),indent=2));print(f'{len(cases)} planned cases -> {dest}; no simulations run')
        else:generate(cfg,a.out,a.limit)
