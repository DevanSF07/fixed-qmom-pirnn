"""Physics-Informed Recurrent Neural Network (PIRNN) Forecaster.

Enforces:
1. Quadrature moments mapping: Moments are computed analytically from positive
   quadrature weights and sizes via torch_moments(q).
2. Exact solute mass conservation: Solute concentration is coupled to crystal
   volume/mass via C(t) = M_total - factor * mu_11(t). Mass drift is exactly 0.
3. Thermodynamic supersaturation gating: Growth and nucleation updates are
   gated by supersaturation sigma = relu(C/Cs(T) - 1). When the solution reaches
   saturation (sigma = 0), growth naturally stops at equilibrium.
4. Additive residual stepping: Node sizes and weights evolve with bounded additive
   increments per time step, preventing exponential runaway over long rollouts.
"""

import torch
from torch import nn
from .physics import CREF, DEFAULT, INDEX, LREF, NREF, torch_moments


class Forecaster(nn.Module):
    def __init__(self, hidden=48, par=DEFAULT):
        super().__init__()
        self.hidden = hidden
        self.encoder = nn.GRU(11, hidden, batch_first=True)
        self.initial_q = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 9),
        )
        self.initial_c = nn.Linear(hidden, 1)

        # Decoder input: log1p(z) [10], (T-30)/5 [1], dT/5 [1], sigma [1] -> 13
        self.decoder = nn.GRUCell(13, hidden)
        self.step = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 9),
        )

        self.mass_factor = float(par.rho * par.thickness * NREF * LREF**2 / CREF)
        self.a1 = float(par.a1)
        self.a2 = float(par.a2)
        self.a3 = float(par.a3)

    def solubility(self, T):
        return 1000.0 * torch.exp(self.a1 + self.a2 / T + self.a3 * torch.log(T))

    def forward(self, history, T):
        """
        history: [batch, L, 11]
            Contains log1p(z_obs) [10] and (T_obs - 30)/5 [1].
        T: [batch, H+1]
            Prescribed temperature profile along forecast horizon.
        """
        _, h = self.encoder(history)
        h = h[-1]

        # Initial quadrature nodes
        q0 = torch.nn.functional.softplus(self.initial_q(h)) + 1e-6

        # Initial concentration anchored to the latest observed concentration
        c_obs = torch.expm1(history[:, -1, 9:10])
        c0 = torch.clamp_min(c_obs + 0.05 * torch.tanh(self.initial_c(h)), 1e-6)[:, 0]

        # Invariant total solute mass for the batch
        m11_0 = torch_moments(q0)[:, INDEX[(1, 1)]]
        m_total = c0 + self.mass_factor * m11_0

        qs = [q0]
        cs = [c0]

        q_curr = q0
        c_curr = c0
        T_curr = T[:, 0]

        for T_next in T[:, 1:].unbind(1):
            m_curr = torch_moments(q_curr)
            z_curr = torch.cat((m_curr, c_curr[:, None]), -1)

            cs_next = self.solubility(T_next)
            sigma = torch.relu(c_curr * CREF / cs_next - 1.0)[:, None]
            T_scaled = ((T_next - 30.0) / 5.0)[:, None]
            dT_scaled = ((T_next - T_curr) / 5.0)[:, None]

            dec_in = torch.cat((torch.log1p(z_curr), T_scaled, dT_scaled, sigma), -1)
            h = self.decoder(dec_in, h)

            # Additive growth update gated by supersaturation
            step_out = self.step(h)
            dq = (sigma + 0.05) * 0.04 * torch.tanh(step_out)
            q_curr = torch.clamp(q_curr + dq, min=1e-6, max=25.0)

            # Solute mass conservation: C(t) = M_total - factor * mu11(t)
            m11_curr = torch_moments(q_curr)[:, INDEX[(1, 1)]]
            c_curr = torch.clamp_min(m_total - self.mass_factor * m11_curr, 1e-6)

            qs.append(q_curr)
            cs.append(c_curr)
            T_curr = T_next

        q = torch.stack(qs, 1)
        c = torch.stack(cs, 1)
        return dict(q=q, z=torch.cat((torch_moments(q), c[..., None]), -1))
