"""Nonlinear crystallization physics: three quadrature nodes, eleven states.

READ THIS FIRST
==============
The physical state contains:
    [w1, w2, w3, L11, L12, L13, L21, L22, L23, concentration, temperature]

There are NINE quadrature quantities, not nine independent physical moments.
We use nine moment equations to determine their nine time derivatives, then
append concentration and prescribed-temperature equations.

Chosen moment orders: 00, 10, 01, 20, 11, 02, 30, 03, 21.
The paper does not enumerate its nine orders: this selection is our assumption.
This is a proposed direct-node quadrature formulation, not a verified exact
reproduction of the authors' DAE implementation. Previous full-range runs of
this assumed formulation encountered numerical failure. Clear comments do not
resolve that issue.

PHYSICAL UNITS AND SCALING
==========================
Physical units: seconds; metres; kg/m^3; degrees Celsius in the supplied
Apelblat correlation. Numerical coefficients are transcribed from the paper;
its coefficient-unit ambiguities have not been experimentally resolved.

The solver stores dimensionless quantities:
    W_i = w_i / NREF                  scaled number concentrations
    X_i = L1_i / LREF                 scaled lengths
    Y_i = L2_i / LREF                 scaled widths
    C   = concentration / CREF        scaled dissolved concentration
Temperature is NOT divided by a reference scale.

A scaled moment is:
    M_pq = sum_i(W_i * X_i**p * Y_i**q)
         = mu_pq / (NREF * LREF**(p+q))

Every sum below runs over nodes i = 1, 2, 3. Time is always seconds.
Growth functions return g1_i = G1_i/LREF, g2_i = G2_i/LREF.
Nucleation returns B_scaled = B_physical/NREF.

THE ELEVEN EQUATIONS
====================
The left sides come from differentiating the quadrature moment definitions.
The right sides come from the population balance and nonlinear growth laws.

 1. sum(dW_i/dt) = B_scaled
 2. sum(X_i*dW_i/dt + W_i*dX_i/dt) = sum(W_i*g1_i)
 3. sum(Y_i*dW_i/dt + W_i*dY_i/dt) = sum(W_i*g2_i)
 4. sum(X_i^2*dW_i/dt + 2*W_i*X_i*dX_i/dt) = 2*sum(W_i*X_i*g1_i)
 5. sum(X_i*Y_i*dW_i/dt + W_i*Y_i*dX_i/dt + W_i*X_i*dY_i/dt)
        = sum(W_i*(Y_i*g1_i + X_i*g2_i))
 6. sum(Y_i^2*dW_i/dt + 2*W_i*Y_i*dY_i/dt) = 2*sum(W_i*Y_i*g2_i)
 7. sum(X_i^3*dW_i/dt + 3*W_i*X_i^2*dX_i/dt) = 3*sum(W_i*X_i^2*g1_i)
 8. sum(Y_i^3*dW_i/dt + 3*W_i*Y_i^2*dY_i/dt) = 3*sum(W_i*Y_i^2*g2_i)
 9. sum(X_i^2*Y_i*dW_i/dt + 2*W_i*X_i*Y_i*dX_i/dt
        + W_i*X_i^2*dY_i/dt) = sum(W_i*(2*X_i*Y_i*g1_i + X_i^2*g2_i))
10. dC/dt = -(rho*h*NREF*LREF^2/CREF) * dM11/dt
11. dT/dt = -cooling_rate(t)

Equations 1--9 form A(q) @ dq_dt = b(q,C,T). They are coupled equations;
their nine unknown derivatives must be solved together.

Model assumptions: well-mixed, constant volume/density/thickness, zero-size
births, no breakage/agglomeration, growth-only (no dissolution). Equation 11
prescribes liquid temperature; it is NOT a complete energy balance.

Run this file to print initial values and rates, NOT to integrate a batch:
    python -m crystallization.physics
or, if this file is outside the package:
    python physics.py
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


# The order here determines the order of moment arrays throughout the project.
# mu21 is the default ninth moment. The older mu12 alternative remains supported
# so existing project configuration/checkpoint interfaces do not break.
ORDERS = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2),
          (3, 0), (0, 3), (2, 1)]
NAMES = [f"mu{p}{q}" for p, q in ORDERS]
INDEX = {order: position for position, order in enumerate(ORDERS)}

# Reference scales are numerical choices, NOT fixed initial conditions.
NREF = 1.0e11       # crystals / m^3 of suspension
LREF = 1.0e-4      # m = 100 micrometres
CREF = 240.0       # kg / m^3
CONDITION_LIMIT = 1.0e10


@dataclass(frozen=True)
class Parameters:
    """Physical coefficients; frozen to prevent accidental modification.

    Gamma must have units m^(-alpha) so gamma*L^alpha is dimensionless.
    kS must carry the compound units implied by B=kS*epsilon*mu11*sigma^beta;
    it cannot generally be assigned the units of B alone.
    """

    k1: float = 1.178e-3       # growth coefficient for length, m/s
    k2: float = 1.874e-4       # growth coefficient for width, m/s
    g1: float = 1.5           # supersaturation exponent for length growth
    g2: float = 1.7           # supersaturation exponent for width growth
    gamma1: float = 0.05      # nonlinear length correction coefficient
    gamma2: float = 0.03      # nonlinear width correction coefficient
    alpha1: float = 0.8       # length exponent; retained as nonlinear
    alpha2: float = 0.9       # width exponent; retained as nonlinear
    kS: float = 3.6e5         # secondary-nucleation coefficient
    epsilon: float = 250.0    # stirring power per mass, W/kg
    birth_exponent: float = 2.0
    rho: float = 1665.0       # crystal density, kg/m^3
    thickness: float = 5e-6   # constant plate thickness h, m
    a1: float = -69.51        # fitted Apelblat coefficients
    a2: float = 368.6
    a3: float = 16.18

    def to_dict(self):
        """Return a serializable parameter dictionary for saved datasets."""
        return asdict(self)


DEFAULT = Parameters()

# With default parameters, scaled total solute is C + MASS_FACTOR*M11.
# state_rates() recalculates this factor when custom parameters are supplied.
MASS_FACTOR = DEFAULT.rho * DEFAULT.thickness * NREF * LREF**2 / CREF


class QuadratureFailure(RuntimeError):
    """Numerical/physical quadrature failure, with details for diagnostics."""

    def __init__(self, message, **details):
        self.details = {"message": message, **details}
        super().__init__(str(self.details))


def set_orders(orders):
    """Select the supported moment set and update its names and lookup table.

    This selects equations; it does not prove that their matrix is nonsingular.
    """
    requested = [tuple(map(int, order)) for order in orders]
    if requested[:-1] != ORDERS[:-1] or requested[-1] not in ((2, 1), (1, 2)):
        raise ValueError("Only the documented mu21 or mu12 sets are supported")

    # Update in place: other project modules may have imported these lists.
    ORDERS[:] = requested
    NAMES[:] = [f"mu{p}{q}" for p, q in ORDERS]
    INDEX.clear()
    INDEX.update({order: position for position, order in enumerate(ORDERS)})


def reference_state():
    """Return Table 2 quadrature in scaled units, with nominal c=240 and T=35.

    Table 2 is used literally: it is not forced to satisfy Table 1's different
    nominal seed mass/mean dimensions. The main block below replaces nominal
    concentration with cs(Th), following saturated hot-solution preparation.
    """
    weights_physical = np.array([1.86e10, 6.46e10, 1.281e11])
    lengths_m = np.array([9.74e-5, 9.80e-5, 1.138e-4])
    widths_m = np.array([6.85e-5, 5.74e-5, 6.05e-5])

    return np.concatenate([
        weights_physical / NREF,
        lengths_m / LREF,
        widths_m / LREF,
        [240.0 / CREF, 35.0],
    ])


def solubility(T, par=DEFAULT):
    """Return physical saturation concentration cs(T), in kg/m^3.

    cs(T) = 1000 * exp(a1 + a2/T + a3*ln(T)).
    T is Celsius for the current fitted-coefficient convention, NOT Kelvin.
    Positive T is required by ln(T), but this mathematical check does not prove
    the correlation's validity at every positive temperature.
    """
    temperature = np.asarray(T, dtype=float)
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0):
        raise ValueError("Finite positive Celsius temperature required")

    return 1000.0 * np.exp(
        par.a1 + par.a2 / temperature + par.a3 * np.log(temperature)
    )


def moments(q, orders=ORDERS):
    """Return SCALED moments from scaled quadrature, for one state or a batch.

    q shape: (..., 9). Output shape: (..., number_of_selected_moments).
    M_pq = sum(W_i * X_i^p * Y_i^q).
    Use physical_moments() afterwards if physical units are needed.
    """
    q = np.asarray(q, dtype=float)
    weights = q[..., :3]
    lengths = q[..., 3:6]
    widths = q[..., 6:9]

    return np.stack([
        np.sum(weights * lengths**p * widths**r, axis=-1)
        for p, r in orders
    ], axis=-1)


def moment_jacobian(q, orders=ORDERS):
    """Build explicit left-hand coefficients A of equations 1--9.

    A = derivative of the selected scaled moments with respect to q.
    Columns: [dW1,dW2,dW3,dX1,dX2,dX3,dY1,dY2,dY3]/dt.
    Consequently dM/dt = A @ dq/dt, by the multivariable chain rule.
    """
    q = np.asarray(q, dtype=float)
    W, X, Y = q[:3], q[3:6], q[6:9]
    zeros = np.zeros(3)

    # Every row below is ordered as [weight terms, length terms, width terms].
    rows = {
        (0, 0): np.r_[np.ones(3), zeros, zeros],       # Equation 1
        (1, 0): np.r_[X, W, zeros],                   # Equation 2
        (0, 1): np.r_[Y, zeros, W],                   # Equation 3
        (2, 0): np.r_[X**2, 2*W*X, zeros],           # Equation 4
        (1, 1): np.r_[X*Y, W*Y, W*X],                # Equation 5
        (0, 2): np.r_[Y**2, zeros, 2*W*Y],           # Equation 6
        (3, 0): np.r_[X**3, 3*W*X**2, zeros],        # Equation 7
        (0, 3): np.r_[Y**3, zeros, 3*W*Y**2],        # Equation 8
        (2, 1): np.r_[X**2*Y, 2*W*X*Y, W*X**2],     # Equation 9, default
        (1, 2): np.r_[X*Y**2, W*Y**2, 2*W*X*Y],     # Alternative equation 9
    }
    return np.stack([rows[tuple(order)] for order in orders])


def growth_birth(q, c, T, par=DEFAULT):
    """Return scaled length growth, width growth, and nucleation rates.

    Inputs: q=[W_i,X_i,Y_i], scaled c=C, Celsius T.
    Physical equations:
        sigma = max(c_physical/cs(T) - 1, 0)
        G1_i = k1*sigma^g1*(1 + gamma1*L1_i^alpha1)
        G2_i = k2*sigma^g2*(1 + gamma2*L2_i^alpha2)
        B = kS*epsilon*mu11*sigma^birth_exponent
    Outputs: G1/LREF, G2/LREF, B/NREF. Time remains in seconds.
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (9,) or not np.isfinite(q).all() or np.any(q <= 0):
        raise QuadratureFailure("Nonpositive/nonfinite quadrature", q=q.tolist())
    if not np.isfinite(c):
        raise QuadratureFailure("Nonfinite concentration", c=float(c))

    W, X, Y = q[:3], q[3:6], q[6:9]
    concentration_physical = c * CREF
    saturation_concentration = float(solubility(T, par))
    sigma = max(concentration_physical / saturation_concentration - 1.0, 0.0)

    # Recover physical sizes BEFORE applying the SI growth coefficients.
    lengths_m = X * LREF
    widths_m = Y * LREF
    G1_physical = par.k1 * sigma**par.g1 * (1 + par.gamma1*lengths_m**par.alpha1)
    G2_physical = par.k2 * sigma**par.g2 * (1 + par.gamma2*widths_m**par.alpha2)

    mu11_physical = NREF * LREF**2 * np.sum(W * X * Y)
    B_physical = par.kS * par.epsilon * mu11_physical * sigma**par.birth_exponent

    return G1_physical / LREF, G2_physical / LREF, B_physical / NREF


def moment_rates(q, c, T, par=DEFAULT, orders=ORDERS):
    """Calculate the explicit RIGHT-HAND sides of the nine moment equations.

    All returned dM_pq/dt are scaled moment derivatives per second. Zero-size
    births contribute directly to M00 only, not the positive-order moments.
    These are moment derivatives, NOT the nine quadrature derivatives dq/dt.
    """
    q = np.asarray(q, dtype=float)
    W, X, Y = q[:3], q[3:6], q[6:9]
    g1, g2, B = growth_birth(q, c, T, par)

    # Equation 1: dM00/dt = B_scaled.
    dM00_dt = B

    # Equation 2: dM10/dt = sum(W_i*g1_i).
    dM10_dt = np.sum(W * g1)

    # Equation 3: dM01/dt = sum(W_i*g2_i).
    dM01_dt = np.sum(W * g2)

    # Equation 4: dM20/dt = 2*sum(W_i*X_i*g1_i).
    dM20_dt = 2 * np.sum(W * X * g1)

    # Equation 5: dM11/dt = sum(W_i*(Y_i*g1_i + X_i*g2_i)).
    dM11_dt = np.sum(W * (Y*g1 + X*g2))

    # Equation 6: dM02/dt = 2*sum(W_i*Y_i*g2_i).
    dM02_dt = 2 * np.sum(W * Y * g2)

    # Equation 7: dM30/dt = 3*sum(W_i*X_i^2*g1_i).
    dM30_dt = 3 * np.sum(W * X**2 * g1)

    # Equation 8: dM03/dt = 3*sum(W_i*Y_i^2*g2_i).
    dM03_dt = 3 * np.sum(W * Y**2 * g2)

    # Equation 9: dM21/dt = sum(W_i*(2*X_i*Y_i*g1_i + X_i^2*g2_i)).
    dM21_dt = np.sum(W * (2*X*Y*g1 + X**2*g2))

    # Compatibility option only: replaces equation 9 if mu12 was selected.
    dM12_dt = np.sum(W * (Y**2*g1 + 2*X*Y*g2))

    rates = {
        (0, 0): dM00_dt, (1, 0): dM10_dt, (0, 1): dM01_dt,
        (2, 0): dM20_dt, (1, 1): dM11_dt, (0, 2): dM02_dt,
        (3, 0): dM30_dt, (0, 3): dM03_dt,
        (2, 1): dM21_dt, (1, 2): dM12_dt,
    }
    return np.array([rates[tuple(order)] for order in orders])


def state_rates(z, cooling_rate, par=DEFAULT, orders=ORDERS,
                condition_limit=CONDITION_LIMIT):
    """Return all ELEVEN state derivatives for the numerical integrator.

    z = [W1,W2,W3,X1,X2,X3,Y1,Y2,Y3,C,T].
    cooling_rate is positive during cooling, in degrees Celsius per second.
    An integrator calls this repeatedly; this function does not advance time.
    """
    z = np.asarray(z, dtype=float)
    if z.shape != (11,) or not np.isfinite(z).all():
        raise QuadratureFailure("An eleven-element finite state is required")

    q = z[:9]
    concentration_scaled = z[9]
    temperature_C = z[10]

    # Equations 1--9: A @ dq_dt = b.
    A = moment_jacobian(q, orders)
    b = moment_rates(q, concentration_scaled, temperature_C, par, orders)
    if A.shape != (9, 9):
        raise ValueError("Exactly nine moment equations are required")

    # Row normalization divides BOTH sides of each equation by the same number.
    # It preserves the solution and reduces differences in equation magnitudes.
    row_norms = np.linalg.norm(A, axis=1)
    if np.any(row_norms == 0) or not np.isfinite(row_norms).all():
        raise QuadratureFailure("Invalid moment-matrix row norms")
    A_scaled = A / row_norms[:, None]
    b_scaled = b / row_norms

    condition = float(np.linalg.cond(A_scaled))
    if not np.isfinite(condition) or condition > condition_limit:
        raise QuadratureFailure("Moment derivative matrix ill-conditioned",
                                condition=condition, limit=condition_limit)
    try:
        # Solve for all nine coupled derivatives; do not explicitly invert A.
        dq_dt = np.linalg.solve(A_scaled, b_scaled)
    except np.linalg.LinAlgError as error:
        raise QuadratureFailure("Singular moment derivative matrix") from error

    # Equation 10, physical: dc/dt = -rho*h*dmu11/dt.
    # Since c=CREF*C and mu11=NREF*LREF^2*M11, the scaled equation is:
    # dC/dt = -(rho*h*NREF*LREF^2/CREF)*dM11/dt.
    dM11_dt = moment_rates(q, concentration_scaled, temperature_C,
                          par, orders=((1, 1),))[0]
    mass_factor = par.rho * par.thickness * NREF * LREF**2 / CREF
    dC_dt = -mass_factor * dM11_dt

    # Equation 11: prescribed cooling, NOT a heat-transfer/energy balance.
    dT_dt = -cooling_rate

    # Ordering matches the initial state and is preserved by the integrator.
    return np.concatenate([dq_dt, [dC_dt, dT_dt]])


def physical_moments(m):
    """Convert selected scaled moments M_pq into physical mu_pq.

    mu_pq = NREF*LREF^(p+q)*M_pq, with units m^(p+q-3)
    when particle count is treated as dimensionless.
    """
    scales = np.array([NREF * LREF**(p+q) for p, q in ORDERS])
    return np.asarray(m) * scales


def torch_moments(q):
    """Same moment definition as moments(), but retain PyTorch gradients.

    The PIRNN predicts positive quadrature quantities; this converts them into
    the moment outputs. q may have shape [batch, time, 9].
    """
    import torch

    W, X, Y = q[..., :3], q[..., 3:6], q[..., 6:9]
    return torch.stack([
        (W * X**p * Y**r).sum(dim=-1) for p, r in ORDERS
    ], dim=-1)


def torch_moment_rates(q, c, T, par=DEFAULT):
    """Differentiable versions of equations 1--9 for the PIRNN physics loss.

    Same scaled quantities and coefficients as moment_rates(). No conversion
    to NumPy: doing that would break the training-gradient calculation.
    No matrix inversion is required here: the loss checks moment equations
    directly along the network's predicted quadrature trajectory.
    """
    import torch

    W, X, Y = q[..., :3], q[..., 3:6], q[..., 6:9]
    cs = 1000 * torch.exp(par.a1 + par.a2/T + par.a3*torch.log(T))
    sigma = torch.relu(c*CREF/cs - 1)

    # [..., None] lets one supersaturation multiply three node growth rates.
    lengths_m = X * LREF
    widths_m = Y * LREF
    g1 = par.k1/LREF * sigma[..., None]**par.g1 * (1 + par.gamma1*lengths_m**par.alpha1)
    g2 = par.k2/LREF * sigma[..., None]**par.g2 * (1 + par.gamma2*widths_m**par.alpha2)
    B = par.kS * par.epsilon * LREF**2 * (W*X*Y).sum(-1) * sigma**par.birth_exponent

    dM00_dt = B                                      # Equation 1
    dM10_dt = (W*g1).sum(-1)                        # Equation 2
    dM01_dt = (W*g2).sum(-1)                        # Equation 3
    dM20_dt = 2*(W*X*g1).sum(-1)                   # Equation 4
    dM11_dt = (W*(Y*g1 + X*g2)).sum(-1)            # Equation 5
    dM02_dt = 2*(W*Y*g2).sum(-1)                   # Equation 6
    dM30_dt = 3*(W*X**2*g1).sum(-1)               # Equation 7
    dM03_dt = 3*(W*Y**2*g2).sum(-1)               # Equation 8
    dM21_dt = (W*(2*X*Y*g1 + X**2*g2)).sum(-1)     # Equation 9
    dM12_dt = (W*(Y**2*g1 + 2*X*Y*g2)).sum(-1)     # Optional replacement for 9

    rates = {
        (0, 0): dM00_dt, (1, 0): dM10_dt, (0, 1): dM01_dt,
        (2, 0): dM20_dt, (1, 1): dM11_dt, (0, 2): dM02_dt,
        (3, 0): dM30_dt, (0, 3): dM03_dt,
        (2, 1): dM21_dt, (1, 2): dM12_dt,
    }
    return torch.stack([rates[tuple(order)] for order in ORDERS], dim=-1)


def load_config(path="config.json"):
    """Read existing project settings and construct its physical parameters."""
    with open(path, encoding="utf-8") as file:
        configuration = json.load(file)
    set_orders(configuration["moments"])
    return configuration, Parameters(**configuration["physics"])


def describe_initial(z, par=DEFAULT):
    """Prepare a readable initial-state report; no time integration occurs.

    Reports the condition number, but does not apply the condition-limit test
    from state_rates(). A finite initial report does not validate a full batch.
    """
    z = np.asarray(z, dtype=float)
    q = z[:9]
    scaled_moments = moments(q)
    A = moment_jacobian(q)
    row_norms = np.linalg.norm(A, axis=1)
    condition = float(np.linalg.cond(A / row_norms[:, None]))
    g1, g2, birth_scaled = growth_birth(q, z[9], z[10], par)
    saturation_concentration = float(solubility(z[10], par))
    concentration_physical = float(z[9] * CREF)

    return {
        "state11_scaled": z.tolist(),
        "moment_orders": ORDERS,
        "moments_physical": physical_moments(scaled_moments).tolist(),
        "T_C": float(z[10]),
        "c_kg_m3": concentration_physical,
        "cs_kg_m3": saturation_concentration,
        "supersaturation": concentration_physical/saturation_concentration - 1,
        "G1_m_s": (g1 * LREF).tolist(),
        "G2_m_s": (g2 * LREF).tolist(),
        "B_number_m3_s": float(birth_scaled * NREF),
        "matrix_row_scaled_condition": condition,
        "crystal_mass_kg_m3": float(
            par.rho * par.thickness * NREF * LREF**2
            * scaled_moments[INDEX[(1, 1)]]
        ),
    }


def main():
    """Print initial diagnostics only; this is the command-line entry point."""
    parser = argparse.ArgumentParser(description="Inspect initial crystallization physics")
    parser.add_argument("--config", default=None, help="Optional existing project config.json")
    parser.add_argument("--hot-temperature", type=float, default=35.2)
    parser.add_argument("--seed-temperature", type=float, default=35.0)
    args = parser.parse_args()

    # Use the existing config if present. The file can also run by itself with
    # the defaults above; an explicitly supplied missing config is an error.
    config_path = args.config
    if config_path is None and Path("config.json").is_file():
        config_path = "config.json"
    if config_path is None:
        par = DEFAULT
    else:
        _, par = load_config(config_path)

    Th = args.hot_temperature
    Ts = args.seed_temperature
    if not (np.isfinite(Th) and np.isfinite(Ts) and Th > Ts > 0):
        parser.error("Require finite hot temperature > seeding temperature > 0 C")

    initial_state = reference_state()
    initial_state[9] = float(solubility(Th, par)) / CREF  # C0 = cs(Th)/CREF
    initial_state[10] = Ts                            # liquid T at seed addition

    report = {
        "initialization": f"Th={Th:g}, Ts={Ts:g}; c0=cs(Th), Table 2 nodes",
        **describe_initial(initial_state, par),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
