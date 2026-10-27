"""Nonlinear quadrature physics, not affine or mean-size residuals.
Trapezoidal residuals have time-discretisation error: test sampling sensitivity.
"""
import torch
from .physics import torch_moment_rates,INDEX,NREF,LREF,CREF,DEFAULT

def losses(pred,target,T,dt,par=DEFAULT):
    z=pred['z'];q=pred['q']
    data=(torch.log(z+1e-8)-torch.log(target+1e-8)).square().mean()
    dm=torch_moment_rates(q,z[...,9],T,par)
    factor=par.rho*par.thickness*NREF*LREF**2/CREF
    f=torch.cat((dm,-factor*dm[...,INDEX[1,1]:INDEX[1,1]+1]),-1)
    residual=z[:,1:]-z[:,:-1]-.5*dt*(f[:,1:]+f[:,:-1])
    scale=(target[:,1:]+target[:,:-1])*.5
    physics=(residual/scale.clamp_min(1e-5)).square().mean()
    mass=z[...,9]+factor*z[...,INDEX[1,1]]
    truth_mass=target[:,0,9]+factor*target[:,0,INDEX[1,1]]
    conservation=((mass-truth_mass[:,None])/truth_mass[:,None]).square().mean()
    return dict(data=data,physics=physics,mass=conservation)
