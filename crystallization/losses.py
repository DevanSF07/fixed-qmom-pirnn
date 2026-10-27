"""Nonlinear quadrature physics, feature-weighted data loss, and mean size ratio losses.
Trapezoidal residuals have time-discretisation error: test sampling sensitivity.
"""
import torch
from .physics import torch_moment_rates, INDEX, NREF, LREF, CREF, DEFAULT

def losses(pred, target, T, dt, par=DEFAULT, mu00_weight=3.0, shape_weight=1.0):
    z = pred['z']
    q = pred['q']
    
    # 1. Feature log squared error with additional weight on mu00
    log_err = (torch.log(torch.clamp_min(z, 1e-8)) - torch.log(torch.clamp_min(target, 1e-8))).square()
    w_feature = torch.ones(10, device=z.device, dtype=z.dtype)
    w_feature[INDEX[0, 0]] = float(mu00_weight)
    data = (log_err * w_feature).mean()

    # 2. Mean length and mean width losses calculated from moment ratios:
    # Mean length = mu10 / mu00; Mean width = mu01 / mu00
    mu00_pred = torch.clamp_min(z[..., INDEX[0, 0]], 1e-8)
    mu10_pred = torch.clamp_min(z[..., INDEX[1, 0]], 1e-8)
    mu01_pred = torch.clamp_min(z[..., INDEX[0, 1]], 1e-8)

    mu00_true = torch.clamp_min(target[..., INDEX[0, 0]], 1e-8)
    mu10_true = torch.clamp_min(target[..., INDEX[1, 0]], 1e-8)
    mu01_true = torch.clamp_min(target[..., INDEX[0, 1]], 1e-8)

    mean_length_pred = mu10_pred / mu00_pred
    mean_length_true = mu10_true / mu00_true

    mean_width_pred = mu01_pred / mu00_pred
    mean_width_true = mu01_true / mu00_true

    loss_length = (torch.log(mean_length_pred) - torch.log(mean_length_true)).square().mean()
    loss_width = (torch.log(mean_width_pred) - torch.log(mean_width_true)).square().mean()
    shape = 0.5 * (loss_length + loss_width)

    # 3. Dynamic moment rate ODE residual physics loss
    dm = torch_moment_rates(q, torch.clamp_min(z[..., 9], 0.0), T, par)
    factor = par.rho * par.thickness * NREF * LREF**2 / CREF
    f = torch.cat((dm, -factor * dm[..., INDEX[1, 1]:INDEX[1, 1] + 1]), -1)
    residual = z[:, 1:] - z[:, :-1] - 0.5 * dt * (f[:, 1:] + f[:, :-1])
    scale = (target[:, 1:] + target[:, :-1]) * 0.5
    physics = (residual / scale.clamp_min(1e-5)).square().mean()

    # 4. Solute mass conservation loss
    mass = z[..., 9] + factor * z[..., INDEX[1, 1]]
    truth_mass = target[:, 0, 9] + factor * target[:, 0, INDEX[1, 1]]
    conservation = ((mass - truth_mass[:, None]) / truth_mass[:, None]).square().mean()

    return dict(
        data=data,
        shape=shape,
        length=loss_length,
        width=loss_width,
        physics=physics,
        mass=conservation
    )
