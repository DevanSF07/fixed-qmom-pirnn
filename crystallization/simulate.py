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
   Continuous handoff between cooling stages: dense output is used to evaluate the
   exact state at stage boundaries (right), ensuring the subsequent stage starts from
   the actual continuous final state rather than an older observation time.
   Temperature handling: Physical kinetics evaluate directly at the prescribed liquid
   temperature T(t) = profile.temperature(t). Numerical temperature integration agrees
   with analytical temperature to machine precision without post-hoc overwriting.
   Positive quadrature (W_i, X_i, Y_i) >= 1e-7 is reconstructed by bounded L-BFGS-B
   optimization with analytical gradient A^T * res. Warm starts are anchored to accepted
   states to guarantee complete independence from solver call order or rejected trial steps.

2. Diagnostic Method ('direct_node'):
   Solves A(q) * dq/dt = b(q, C, T) directly in coordinate space. Retained as an analytical
   diagnostic demonstrating coordinate-inversion failure under active nucleation.
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


def deterministic_prior(m_target):
    """
    Construct a deterministic initial quadrature guess q = [W1..3, X1..3, Y1..3]
    purely from the target moments, ensuring independent, repeatable optimization.
    """
    m = np.asarray(m_target, dtype=float)
    w_sum = m[0]
    l_mean = m[1] / max(m[0], 1e-12)
    w_mean = m[2] / max(m[0], 1e-12)
    l_var = max(m[3] / max(m[0], 1e-12) - l_mean**2, 1e-10)
    w_var = max(m[5] / max(m[0], 1e-12) - w_mean**2, 1e-10)
    l_std = np.sqrt(l_var)
    w_std = np.sqrt(w_var)

    w = np.array([w_sum * 0.25, w_sum * 0.50, w_sum * 0.25])
    x = np.array([max(l_mean - 0.8 * l_std, 1e-4), l_mean, l_mean + 0.8 * l_std])
    y = np.array([max(w_mean - 0.8 * w_std, 1e-4), w_mean, w_mean + 0.8 * w_std])
    return np.r_[w, x, y]


def reconstruct_quadrature(m_target, q_guess=None, maxiter=300, gtol=1e-8, ftol=1e-15):
    """
    Reconstruct positive 3-node quadrature q = [W1..3, X1..3, Y1..3] from 9 target moments.

    Solves:
        min_q  sum_k ((M_k(q) - M_k,target) / (M_k,target + 1e-12))^2
        s.t.   q_i >= 1e-7  for all i=1..9

    Analytical gradient:
        grad_q = 2 * A(q)^T @ ((M(q) - M_target) / (M_target + 1e-12)^2)
    """
    m_target = np.asarray(m_target, dtype=float)
    if m_target.shape != (9,) or not np.isfinite(m_target).all() or np.any(m_target <= 0):
        raise QuadratureFailure("Invalid nonpositive or nonfinite moment target", moments=m_target.tolist())

    denom = np.maximum(np.abs(m_target), 1e-12)

    def obj(q):
        m_calc = moments(q)
        return float(np.sum(((m_calc - m_target) / denom) ** 2))

    def grad(q):
        m_calc = moments(q)
        res = 2.0 * (m_calc - m_target) / (denom ** 2)
        A = moment_jacobian(q)
        return A.T @ res

    bounds = [(1e-7, None)] * 9

    # Use supplied initial guess if finite and positive; otherwise use deterministic prior
    if q_guess is not None and np.isfinite(q_guess).all() and np.all(q_guess > 0):
        initial_guess = np.asarray(q_guess, dtype=float)
    else:
        initial_guess = deterministic_prior(m_target)

    res = minimize(
        obj,
        initial_guess,
        jac=grad,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol},
    )

    if not np.isfinite(res.x).all():
        raise QuadratureFailure("Optimization returned nonfinite quadrature", status=res.status, message=res.message)

    m_rec = moments(res.x)
    rel_errors = np.abs(m_rec - m_target) / denom
    max_err = float(np.max(rel_errors))
    mean_err = float(np.mean(rel_errors))

    best_x = res.x
    best_max_err = max_err
    best_mean_err = mean_err
    best_ok = res.success
    best_msg = res.message

    # Multi-start refinement if relative error exceeds 0.5%
    if max_err > 0.005:
        # Candidate 2: deterministic bimodal split (fine nuclei + intermediate + seed)
        w_sum = m_target[0]
        l_mean = m_target[1] / max(m_target[0], 1e-12)
        w_mean = m_target[2] / max(m_target[0], 1e-12)
        g_refine = np.r_[
            w_sum * 0.6, w_sum * 0.25, w_sum * 0.15,
            l_mean * 0.2, l_mean * 1.0, l_mean * 1.8,
            w_mean * 0.2, w_mean * 1.0, w_mean * 1.8,
        ]
        res2 = minimize(
            obj,
            g_refine,
            jac=grad,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol},
        )
        if np.isfinite(res2.x).all():
            errs2 = np.abs(moments(res2.x) - m_target) / denom
            max_err2 = float(np.max(errs2))
            if max_err2 < best_max_err:
                best_x = res2.x
                best_max_err = max_err2
                best_mean_err = float(np.mean(errs2))
                best_ok = res2.success
                best_msg = res2.message

    if not np.isfinite(best_max_err):
        raise QuadratureFailure("Nonfinite reconstruction error", status=best_ok, message=best_msg)

    # Canonicalize node ordering by crystal length X (X1 <= X2 <= X3)
    # Moving linked triples [W, X, Y] together
    order = np.argsort(best_x[3:6])
    q_canonical = np.r_[best_x[:3][order], best_x[3:6][order], best_x[6:9][order]]
    return q_canonical, best_max_err, best_mean_err, best_ok, best_msg


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
        'moment_reconstruction' (production mode) or 'direct_node' (diagnostic mode).
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
        Dictionary containing solver diagnostics.
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
        # DIAGNOSTIC SOLVER: A(q) * dq/dt = b(q, C, T)
        states = []
        last = z0.copy()
        maxcond = 0.0

        for left, right in zip(boundaries[:-1], boundaries[1:]):
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
            if np.any(mask):
                states.extend(sol.sol(times[mask]).T)
            # Continuous handoff at stage boundary
            last = sol.sol(right)

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
    q0, err0, _, _, _ = reconstruct_quadrature(moments(z0[:9]), z0[:9])
    m0 = moments(q0)
    y0 = np.r_[m0, z0[9], z0[10]]

    # Fixed accepted anchor: initialized at t=0, updated ONLY at verified segment boundaries
    anchor_q = q0.copy()
    max_rhs_err = 0.0
    rhs_eval_count = 0

    all_sol_m = []
    all_sol_c = []
    all_sol_T = []

    last_y = y0.copy()

    for left, right in zip(boundaries[:-1], boundaries[1:]):
        def rhs_moments(t, y):
            nonlocal max_rhs_err, rhs_eval_count
            rhs_eval_count += 1
            m = y[:9]
            c = y[9]

            # True physical liquid temperature at time t (no drift, no overwriting)
            T_true = float(profile.temperature(t))

            # Controlled reconstruction using fixed accepted anchor from segment start
            # No mutable anchor update occurs during trial steps!
            q, max_err, _, _, msg = reconstruct_quadrature(m, anchor_q)
            max_rhs_err = max(max_rhs_err, max_err)

            if max_err > max_reconstruction_error:
                raise QuadratureFailure(
                    f"RHS reconstruction error ({max_err:.4e}) exceeded limit ({max_reconstruction_error})",
                    time_s=float(t),
                    reconstruction_error=max_err,
                    moments=m.tolist(),
                    optimizer_message=msg,
                )

            # Physical kinetics evaluated at exact liquid temperature
            dm = moment_rates(q, c, T_true, par)
            dC = -mass_factor * dm[INDEX[(1, 1)]]

            rate_time = min(t, np.nextafter(right, left))
            dT = -float(profile.rate(rate_time))

            return np.r_[dm, dC, dT]

        # Use dense_output=True so we evaluate at observation points AND get exact boundary state
        sol = solve_ivp(
            fun=rhs_moments,
            t_span=(left, right),
            y0=last_y,
            method=method,
            rtol=rtol,
            atol=atol,
            dense_output=True,
            max_step=min(max_step_s, right - left),
        )

        if not sol.success:
            raise QuadratureFailure(
                "Integrator failed in moment_reconstruction mode",
                time_s=float(sol.t[-1]),
                solver_message=sol.message,
            )

        mask = (times >= left) & (times <= right) if left == 0 else (times > left) & (times <= right)
        t_eval_segment = times[mask]
        if len(t_eval_segment) > 0:
            sol_eval = sol.sol(t_eval_segment)
            all_sol_m.append(sol_eval[:9].T)
            all_sol_c.append(sol_eval[9])
            all_sol_T.append(sol_eval[10])

        # EXACT CONTINUOUS HANDOFF AT right:
        # Evaluates the actual final state at right to begin the next stage
        last_y = sol.sol(right)

        # Update anchor_q ONLY at the verified, accepted boundary right
        q_right, err_right, _, _, msg_right = reconstruct_quadrature(last_y[:9], anchor_q)
        if err_right > max_reconstruction_error:
            raise QuadratureFailure(
                f"Boundary reconstruction error ({err_right:.4e}) exceeded limit ({max_reconstruction_error})",
                time_s=float(right),
                reconstruction_error=err_right,
                optimizer_message=msg_right,
            )
        anchor_q = q_right.copy()

    m = np.vstack(all_sol_m)
    c = np.concatenate(all_sol_c)
    T = np.concatenate(all_sol_T)

    # Reconstruct positive quadrature at all observation times
    q_recorded = []
    rec_errors = []
    prev_accepted_q = q0.copy()

    for m_target in m:
        q_val, max_err, mean_err, ok, msg = reconstruct_quadrature(m_target, prev_accepted_q)
        if max_err > max_reconstruction_error:
            raise QuadratureFailure(
                f"Observation quadrature reconstruction error ({max_err:.4e}) exceeded limit",
                reconstruction_error=max_err,
                optimizer_message=msg,
            )
        q_recorded.append(q_val)
        rec_errors.append(max_err)
        prev_accepted_q = q_val.copy()

    q_array = np.array(q_recorded)
    z = np.column_stack((q_array, c, T))

    # Physical validity checks
    if not np.isfinite(z).all() or np.any(z[:, :10] <= 0):
        raise QuadratureFailure("Invalid nonpositive or nonfinite integrated states")

    # Verify temperature agreement with prescribed curve WITHOUT overwriting
    expected_T = profile.temperature(times)
    T_err = float(np.max(np.abs(z[:, 10] - expected_T)))
    if T_err > 0.002:
        raise QuadratureFailure("Temperature integration mismatch", max_error_C=T_err)

    # Verify solute mass conservation: C + mass_factor * M11
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