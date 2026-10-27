"""PIRNN only. No baseline/comparison experiment. No future noisy targets.
Validation error selects a checkpoint; clean test data are never used in fitting.
"""
import argparse, hashlib, json, random, subprocess
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .physics import load_config
from .data import Windows
from .model import Forecaster
from .losses import losses

def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return "unknown"

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def train(data='outputs/data',out='outputs/pirnn',config='config.json',epochs=None):
    cfg, par = load_config(config)
    s = cfg['training']
    epochs = epochs if epochs is not None else s['epochs']
    if epochs < 1:
        raise ValueError('epochs must be positive')
    seed = cfg['random_seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(4)
    tr = Windows(data, 'train', s['history'], s['horizon'])
    va = Windows(data, 'val', s['history'], s['horizon'])
    if cfg['physics'] != tr.cfg['physics'] or cfg['moments'] != tr.cfg['moments']:
        raise ValueError('Dataset physics differs from training configuration')
    if cfg['noise'] != tr.cfg['noise']:
        raise ValueError('Noise configuration differs from saved observations')
    if not np.isclose(tr.dt, cfg['solver']['sample_interval_s']):
        raise ValueError('Sampling interval mismatch')
    if tr.cfg['physics'] != va.cfg['physics']:
        raise ValueError('Validation parameters differ')

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(data) / 'manifest.json'
    manifest_hash = sha256_file(manifest_path)
    git_commit = get_git_commit()

    model = Forecaster(s['hidden'], par=par)
    opt = torch.optim.Adam(model.parameters(), lr=s['learning_rate'])
    logs = []
    best = float('inf')
    failures = len(json.loads(manifest_path.read_text())['failed'])
    print(f'Train windows={len(tr)}, val windows={len(va)}, failed batches={failures}. Metrics cover successful cases only.')

    arch_info = {
        "model_class": "crystallization.model.Forecaster",
        "hidden": s['hidden'],
        "encoder": "nn.GRU(11, hidden, batch_first=True)",
        "initial_q": "nn.Sequential(Linear(hidden, hidden), ReLU(), Linear(hidden, 9))",
        "initial_c": "nn.Linear(hidden, 1)",
        "decoder": "nn.GRUCell(13, hidden)",
        "step": "nn.Sequential(Linear(hidden, hidden), Tanh(), Linear(hidden, 9))",
        "keys": list(model.state_dict().keys()),
    }

    mu00_w = float(s.get('mu00_weight', 3.0))
    shape_w = float(s.get('shape_weight', 1.0))

    for epoch in range(epochs):
        model.train()
        sums = dict(data=0., shape=0., physics=0., mass=0.)
        count = 0
        for hist, T, y in DataLoader(tr, batch_size=s['batch_size'], shuffle=True):
            opt.zero_grad()
            terms = losses(model(hist, T), y, T, tr.dt, par, mu00_weight=mu00_w, shape_weight=shape_w)
            loss = terms['data'] + shape_w * terms['shape'] + s['physics_weight'] * terms['physics'] + s['mass_weight'] * terms['mass']
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), s['gradient_clip'])
            opt.step()
            for key in sums:
                sums[key] += terms[key].item() * len(hist)
            count += len(hist)

        model.eval()
        value = 0.
        n = 0
        with torch.no_grad():
            for hist, T, y in DataLoader(va, batch_size=s['batch_size']):
                val_terms = losses(model(hist, T), y, T, va.dt, par, mu00_weight=mu00_w, shape_weight=shape_w)
                val_loss = val_terms['data'] + shape_w * val_terms['shape']
                value += val_loss.item() * len(hist)
                n += len(hist)

        val_score = value / n
        row = dict(epoch=epoch + 1, **{k: v / count for k, v in sums.items()}, val_loss=val_score)
        logs.append(row)
        print(json.dumps(row), flush=True)

        if val_score < best:
            best = val_score
            checkpoint_data = {
                "model": model.state_dict(),
                "config": cfg,
                "epochs_requested": epochs,
                "best_epoch": epoch + 1,
                "val_log_mse": best,
                "data_manifest": str(manifest_path),
                "data_manifest_sha256": manifest_hash,
                "git_commit": git_commit,
                "architecture": arch_info,
            }
            torch.save(checkpoint_data, out / 'best.pt')

        (out / 'training.json').write_text(json.dumps(logs, indent=2))
    return logs

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='config.json');p.add_argument('--data',default='outputs/data');p.add_argument('--out',default='outputs/pirnn');p.add_argument('--epochs',type=int)
    train(**vars(p.parse_args()))
