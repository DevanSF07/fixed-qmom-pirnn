"""GRU with positive quadrature decoder; outputs moments through quadrature.
Network input is log1p(scaled moments/concentration) and scaled temperature.
Both baseline and PIRNN use the same architecture. No future truth is fed back.
"""
import torch
from torch import nn
from .physics import torch_moments
class Forecaster(nn.Module):
    def __init__(self,hidden=48):
        super().__init__();self.hidden=hidden
        self.encoder=nn.GRU(11,hidden,batch_first=True)
        self.initial=nn.Linear(hidden,10)
        self.decoder=nn.GRUCell(11,hidden)
        self.step=nn.Linear(hidden,10)
    def forward(self,history,T):
        _,h=self.encoder(history);h=h[-1]
        state=torch.nn.functional.softplus(self.initial(h))+1e-6
        qs=[state[:,:9]];cs=[state[:,9]]
        for temperature in T[:,1:].unbind(1):
            z=torch.cat((torch_moments(state[:,:9]),state[:,9:]),-1)
            h=self.decoder(torch.cat((torch.log1p(z),(temperature[:,None]-30)/5),-1),h)
            state=state*torch.exp(.1*torch.tanh(self.step(h)))
            qs.append(state[:,:9]);cs.append(state[:,9])
        q=torch.stack(qs,1);c=torch.stack(cs,1)
        return dict(q=q,z=torch.cat((torch_moments(q),c[...,None]),-1))
