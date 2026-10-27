"""
Simulate crystallization batches using the nonlinear QMOM formulation.

Mathematical Foundation
-----------------------
Paper: Szilágyi and Lakatos (2015), Periodica Polytechnica Chemical Engineering.
DOI: 10.3311/PPch.7581

The physical system evolves plate-like crystals with length L1 and width L2,
constant thickness h, size-dependent nonlinear growth (G1, G2), active secondary
nucleation (B), solute mass conservation (c), and prescribed cooling (T).

Nine Moment Equations (Orders: 00, 10, 01, 20, 11, 02, 30, 03, 21):
    dM00/dt = B_scaled
    dM10/dt = sum_i(W_i * g1_i)
    dM01/dt = sum_i(W_i * g2_i)
    dM20/dt = 2 * sum_i(W_i * X_i * g1_i)
    dM11/dt = sum_i(W_i * (Y_i * g1_i + X_i * g2_i))
    dM02/dt = 2 * sum_i(W_i * Y_i * g2_i)
    dM30/dt = 3 * sum_i(W_i * X_i^2 * g1_i)
    dM03/dt = 3 * sum_i(W_i * Y_i^2 * g2_i)
    dM21/dt = sum_i(W_i * (2 * X_i * Y_i * g1_i + X_i^2 * g2_i))

Solute mass balance:
    dc/dt = -rho * thickness * dmu11/dt
    dC/dt = -(rho * thickness * NREF * LREF^2 / CREF) * dM11/dt

Prescribed cooling:
    dT/dt = -cooling_rate(t)

Numerical Architecture
----------------------
1. Production Method ('moment_reconstruction'):
   Integrates the 9 moments + C + T directly:
       dy/dt = [dM/dt, dC/dt, dT/dt]
   At every internal solver stage (RHS call), positive quadrature (W_i, X_i, Y_i) >= 1e-7
   is reconstructed by bounded optimization (L-BFGS-B with analytical gradient A^T * res).
   Warm starts are anchored strictly to monotonically advancing accepted times to ensure
   reproducibility independent of solver call order or rejected trial steps.
   Reconstruction residuals are strictly checked across all 9 moments.

2. Diagnostic Method ('direct_node'):
   Solves A(q) * dq/dt = b(q, C, T) directly in coordinate space. Under active nucleation
   (kS = 3.6e5), zero-size births enter dM00/dt only, forcing the coordinate velocity dq/dt
   negative for small crystal nodes, driving length coordinates through zero into negative
   unphysical values near t ~ 608 s. Retained strictly as an analytical diagnostic.
"""

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from .physics import (
    CREF,
    DEFAULT,
    INDEX,
    LREF,
    NREF,
    ORDERS,
    Parameters,
    QuadratureFailure,
    load_config,
    moment_jacobian,
    moment_rates,
    moments,
    state_rates,
)
from .profiles import from_metadata


def reconstruct_quadrature(m_target, q_guess, maxiter=250, gtol=1e-7, ftol=1e-14):
    """
    Reconstruct positive 3-node quadrature q = [W1..3, X1..3, Y1..3] from 9 target moments.

    Solves:
        min_q  sum_k ((M_k(q) - M_k,target) / (M_k,target + 1e-12))^2
        s.t.   q_i >= 1e-7  for all i=1..9

    Analytical gradient:
        grad_q = 2 * A(q)^T @ ((M(q) - M_target) / (M_target + 1e-12)^2)
    where A(q) = dM/dq is the analytical moment Jacobian.
    """
    m_target = np.asarray(m_target, dtype=float)
    denom = m_target + 1e-12

    def obj(q):
        m_calc = moments(q)
        return float(np.sum(((m_calc - m_target) / denom) ** 2))

    def grad(q):
        m_calc = moments(q)
        res = 2.0 * (m_calc - m_target) / (denom ** 2)
        A = moment_jacobian(q)
        return A.T @ res

    bounds = [(1e-7, None)] * 9

    res = minimize(
        obj,
        q_guess,
        jac=grad,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol},
    )

    m_reconstructed = moments(res.x)
    rel_errors = np.abs(m_reconstructed - m_target) / denom
    max_rel_error = float(np.max(rel_errors))
    mean_rel_error = float(np.mean(rel_errors))

    return res.x, max_rel_error, mean_rel_error, res.success


def simulate(
    z0,
    profile,
    times,
    par=DEFAULT,
    rtol=1e-6,
    atol=1e-9,
    method="RK45",
    solver_mode="moment_reconstruction",
    condition_limit=1e11,
    max_step_s=60.0,
    max_reconstruction_error=0.05,
):
    """
    Integrate the eleven crystallization equations for one batch.

    Inputs
    ------
    z0:
        Initial SCALED state [W1..3, X1..3, Y1..3, C, T].
    profile:
        Cooling profile object (rate, temperature, breakpoints).
    times:
        Observation recording times [0, dt, 2*dt, ...].
    par:
        Physical Parameters dataclass.
    rtol, atol:
        ODE solver tolerances.
    method:
        Integration scheme ('RK45', 'Radau', etc.).
    solver_mode:
        'moment_reconstruction' (default production mode: integrates moments,
         reconstructs positive quadrature via bounded optimization).
        'direct_node' (diagnostic mode: solves A(q) dq/dt = b in coordinate space).
    condition_limit:
        Condition number limit for direct_node Jacobian.
    max_step_s:
        Maximum solver internal time step.
    max_reconstruction_error:
        Maximum permitted relative reconstruction error across all 9 moments (default 0.05).

    Returns
    -------
    z:
        Recorded eleven-variable state trajectory [N, 11].
    m:
        Recorded nine scaled moments [N, 9].
    diagnostics:
        Dictionary containing solver diagnostics (mass drift, reconstruction residuals, etc.).
    """
    times = np.asarray(times, dtype=float)
    if (
        times.ndim != 1
        or len(times) < 2
        or times[0] != 0
        or np.any(np.diff(times) <= 0)
        or times[-1] > profile.duration + 1e-8
    ):
        raise ValueError("Invalid observation times")

    z0 = np.asarray(z0, dtype=float).copy()
    if z0.shape != (11,) or abs(z0[10] - profile.start) > 1e-6:
        raise ValueError("Initial temperature/state mismatch")

    mass_factor = par.rho * par.thickness * NREF * LREF**2 / CREF

    boundaries = np.unique(
        np.r_[
            0,
            profile.breakpoints[profile.breakpoints < times[-1]],
            times[-1],
        ]
    )

    if solver_mode == "direct_node":
        states = []
        last = z0.copy()
        maxcond = 0.0

        for left, right in zip(boundaries[:-1], boundaries[1:]):
            last[10] = float(profile.temperature(left))
            def rhs_direct(t, z):
                nonlocal maxcond
                try:
                    rate_time = min(t, np.nextafter(right, left))
                    cooling_rate = float(profile.rate(rate_time))
                    rates = state_rates(
                        z,
                        cooling_rate,
                        par,
                        condition_limit=condition_limit,
                    )
                    A = moment_jacobian(z[:9])
                    row_norms = np.linalg.norm(A, axis=1)
                    A_scaled = A / row_norms[:, None]
                    maxcond = max(maxcond, float(np.linalg.cond(A_scaled)))
                    return rates
                except QuadratureFailure as error:
                    raise QuadratureFailure(
                        "Simulation stopped in direct_node mode",
                        time_s=float(t),
                        cause=error.details,
                    ) from error

            sol = solve_ivp(
                fun=rhs_direct,
                t_span=(left, right),
                y0=last,
                method=method,
                rtol=rtol,
                atol=atol,
                dense_output=True,
                max_step=min(max_step_s, right - left),
            )

            if not sol.success:
                raise QuadratureFailure(
                    "Integrator failed in direct_node mode",
                    time_s=float(sol.t[-1]),
                    solver_message=sol.message,
                )

            mask = (times >= left) & (times <= right) if left == 0 else (times > left) & (times <= right)
            states.extend(sol.sol(times[mask]).T)
            last = sol.y[:, -1]

        z = np.array(states)
        m = moments(z[:, :9])
        mass = z[:, 9] + mass_factor * m[:, INDEX[(1, 1)]]
        mass_drift = float(np.ptp(mass))

        return z, m, {
            "solver_mode": "direct_node",
            "max_row_scaled_condition": maxcond,
            "mass_drift_scaled": mass_drift,
        }

    # PRODUCTION SOLVER: MOMENT INTEGRATION + QUADRATURE RECONSTRUCTION
    q0 = z0[:9].copy()
    m0 = moments(q0)
    y0 = np.r_[m0, z0[9], z0[10]]

    anchor = {"t": 0.0, "q": q0.copy()}
    max_rhs_err = 0.0
    rhs_eval_count = 0

    all_sol_m = []
    all_sol_c = []
    all_sol_T = []

    last_y = y0.copy()

    for left, right in zip(boundaries[:-1], boundaries[1:]):
        last_y[10] = float(profile.temperature(left))
        def rhs_moments(t, y):
            nonlocal max_rhs_err, rhs_eval_count
            rhs_eval_count += 1
            m = y[:9]
            c = y[9]
            T = y[10]

            q, max_err, _, _ = reconstruct_quadrature(m, anchor["q"])
            max_rhs_err = max(max_rhs_err, max_err)

            if max_err > max_reconstruction_error:
                raise QuadratureFailure(
                    f"Reconstruction error ({max_err:.4e}) exceeded limit ({max_reconstruction_error})",
                    time_s=float(t),
                    reconstruction_error=max_err,
                    moments=m.tolist(),
                )

            if t > anchor["t"]:
                anchor["t"] = t
                anchor["q"] = q.copy()

            dm = moment_rates(q, c, T, par)
            dC = -mass_factor * dm[INDEX[(1, 1)]]

            rate_time = min(t, np.nextafter(right, left))
            dT = -float(profile.rate(rate_time))

            return np.r_[dm, dC, dT]

        mask = (times >= left) & (times <= right) if left == 0 else (times > left) & (times <= right)
        t_eval_segment = times[mask]

        sol = solve_ivp(
            fun=rhs_moments,
            t_span=(left, right),
            y0=last_y,
            method=method,
            rtol=rtol,
            atol=atol,
            t_eval=t_eval_segment,
            max_step=min(max_step_s, right - left),
        )

        if not sol.success:
            raise QuadratureFailure(
                "Integrator failed in moment_reconstruction mode",
                time_s=float(sol.t[-1]),
                solver_message=sol.message,
            )

        if len(t_eval_segment) > 0:
            all_sol_m.append(sol.y[:9].T)
            all_sol_c.append(sol.y[9])
            all_sol_T.append(sol.y[10])

        last_y = sol.y[:, -1]

    m = np.vstack(all_sol_m)
    c = np.concatenate(all_sol_c)
    T = np.concatenate(all_sol_T)

    q_recorded = []
    rec_errors = []
    q_guess = q0.copy()

    for m_target in m:
        q_val, max_err, mean_err, _ = reconstruct_quadrature(m_target, q_guess)
        if max_err > max_reconstruction_error:
            raise QuadratureFailure(
                f"Observation quadrature reconstruction error ({max_err:.4e}) exceeded limit",
                reconstruction_error=max_err,
            )
        q_recorded.append(q_val)
        rec_errors.append(max_err)
        q_guess = q_val.copy()

    q_array = np.array(q_recorded)
    z = np.column_stack((q_array, c, T))

    if not np.isfinite(z).all() or np.any(z[:, :10] <= 0):
        raise QuadratureFailure("Invalid nonpositive or nonfinite integrated states")

    expected_T = profile.temperature(times)
    T_err = float(np.max(np.abs(z[:, 10] - expected_T)))
    if T_err > 0.05:
        raise QuadratureFailure("Temperature integration mismatch", max_error_C=T_err)
    z[:, 10] = expected_T

    mass = z[:, 9] + mass_factor * m[:, INDEX[(1, 1)]]
    mass_drift = float(np.ptp(mass))
    allowed_drift = 1e-5 * max(abs(mass[0]), 1.0)
    if mass_drift > allowed_drift:
        raise QuadratureFailure("Mass conservation failed", drift=mass_drift, allowed=allowed_drift)

    diagnostics = {
        "solver_mode": "moment_reconstruction",
        "max_rhs_reconstruction_error": float(max_rhs_err),
        "max_observation_reconstruction_error": float(np.max(rec_errors)),
        "mean_observation_reconstruction_error": float(np.mean(rec_errors)),
        "mass_drift_scaled": mass_drift,
        "temperature_max_error_C": T_err,
        "rhs_evaluations": rhs_eval_count,
    }

    return z, m, diagnostics


def run_case(case, cfg, out):
    """
    Simulate one dataset case and save its complete clean trajectory.
    """
    par = Parameters(**cfg["physics"])
    profile = from_metadata(case["profile"])
    z0 = np.array(case["initial_state11_scaled"])

    opts = cfg["solver"]
    dt = opts["sample_interval_s"]

    n = int(round(profile.duration / dt))
    times = np.arange(n + 1, dtype=float) * dt

    try:
        z, m, diagnostics = simulate(
            z0=z0,
            profile=profile,
            times=times,
            par=par,
            rtol=opts.get("rtol", 1e-6),
            atol=opts.get("atol", 1e-9),
            method=opts.get("method", "RK45"),
            solver_mode=opts.get("solver_mode", "moment_reconstruction"),
            condition_limit=opts.get("condition_limit", 1e11),
            max_step_s=opts.get("max_step_s", 60.0),
        )
    except QuadratureFailure as error:
        path = Path(out).with_suffix(".failure.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "case": case,
                    "failure": error.details,
                },
                indent=2,
            )
        )
        raise

    clean = np.column_stack((m, z[:, 9]))
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,
        t=times,
        state11=z,
        q=z[:, :9],
        clean=clean,
        T=z[:, 10],
        orders=ORDERS,
        parameters=json.dumps(par.to_dict()),
        case=json.dumps(case),
        diagnostics=json.dumps(diagnostics),
    )
    return path


if __name__ == "__main__":
    import argparse
    from .data import make_design

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json", help="Configuration file")
    parser.add_argument("--case", type=int, default=0, help="Index of the batch to simulate")
    parser.add_argument("--out", default="outputs/single_batch.npz", help="Destination path")
    parser.add_argument(
        "--solver-mode",
        choices=["moment_reconstruction", "direct_node"],
        default="moment_reconstruction",
        help="Solver formulation: moment_reconstruction (production) or direct_node (diagnostic)",
    )

    args = parser.parse_args()
    cfg, _ = load_config(args.config)
    cfg["solver"]["solver_mode"] = args.solver_mode

    cases = make_design(cfg)
    if not 0 <= args.case < len(cases):
        parser.error(f"Case index {args.case} outside design (total cases: {len(cases)})")

    try:
        saved_path = run_case(
            case=cases[args.case],
            cfg=cfg,
            out=args.out,
        )
        print("SIMULATION SUCCESS:", saved_path)
    except QuadratureFailure as error:
        parser.exit(1, f"Simulation failed: {error}\n")