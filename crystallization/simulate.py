"""
Simulate one crystallization batch using the nonlinear 11-variable model.

The state contains:
    3 quadrature weights
    3 length nodes
    3 width nodes
    1 dissolved concentration
    1 liquid temperature

The ODE solver evolves these variables through time.
The nine moments are then calculated from the quadrature variables.

If the quadrature becomes invalid or numerically unsuitable, the simulation
stops. A failed simulation is not saved as a complete training trajectory.
"""

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from .physics import *
from .profiles import from_metadata


def simulate(
    z0,
    profile,
    times,
    par=DEFAULT,
    rtol=1e-8,
    atol=1e-11,
    condition_limit=CONDITION_LIMIT,
    method="Radau",
    max_step_s=60.0,
):
    """
    Integrate the eleven equations for one batch.

    Inputs
    ------
    z0:
        Initial SCALED state, ordered as:
        [W1, W2, W3, X1, X2, X3, Y1, Y2, Y3, C, T]

    profile:
        Cooling profile object. It supplies:
            profile.rate(t)        -> positive cooling rate
            profile.temperature(t) -> prescribed liquid temperature
            profile.breakpoints    -> times separating profile sections

    times:
        Times at which we want to RECORD the solution.


    par:
        Physical and kinetic parameters.

    rtol, atol:
        Relative and absolute tolerances used by the ODE solver.

    condition_limit:
        Maximum permitted condition number of the row-scaled moment matrix.

    method:
        ODE solver method. The default is Radau, an implicit method commonly
        used for stiff differential equations.

    max_step_s:
        Maximum permitted internal solver step, in seconds.
        The solver may choose much smaller steps.

    Returns
    -------
    z:
        Recorded eleven-variable states. Shape: [number_of_times, 11].

    m:
        Nine SCALED moments at the recorded times.
        Shape: [number_of_times, 9].

    diagnostics:
        Largest encountered matrix condition number and scaled mass drift.
    """

    # --------------------------------------------------------------
    # 1. Check the requested observation times.
    # --------------------------------------------------------------

    times = np.asarray(times, dtype=float)

    if (
        times.ndim != 1
        or len(times) < 2
        or times[0] != 0
        or np.any(np.diff(times) <= 0)
        or times[-1] > profile.duration + 1e-8
    ):
        raise ValueError("Invalid observation times")

    # --------------------------------------------------------------
    # 2. Prepare and check the initial state.
    # --------------------------------------------------------------

    # Make a copy so that this function does not modify the supplied array.
    z0 = np.asarray(z0, dtype=float).copy()

    # The state must contain eleven values.
    # Its initial temperature must match the cooling profile's start.
    if (
        z0.shape != (11,)
        or abs(z0[10] - profile.start) > 1e-9
    ):
        raise ValueError("Initial temperature/state mismatch")

    # Recorded states will be collected here.
    states = []

    # 'last' is the state from which the next integration segment begins.
    last = z0.copy()

    # Track the largest matrix condition number encountered by the solver.
    maxcond = 0.0

    # --------------------------------------------------------------
    # 3. Divide the cooling profile into integration segments.
    # --------------------------------------------------------------

    # Some profiles have abrupt changes in cooling rate:
    # for example, the start and end of a hold.
    #
    # Integrating each section separately prevents the solver from stepping
    # across these changes as if the derivative were smooth.
    #
    # Example boundaries:
    #     [0, hold_start_time, hold_end_time, final_time]
    #
    # Only include breakpoints before the final requested observation time.
    boundaries = np.unique(
        np.r_[
            0,
            profile.breakpoints[
                profile.breakpoints < times[-1]
            ],
            times[-1],
        ]
    )

    # Integrate one segment at a time.
    for left, right in zip(boundaries[:-1], boundaries[1:]):

        # ----------------------------------------------------------
        # 4. Define the derivative function required by solve_ivp.
        # ----------------------------------------------------------

        def rhs(t, z):
            """
            Return dz/dt at the solver's current time and trial state.

            solve_ivp calls this function repeatedly.
            This function calculates derivatives; it does not advance time.
            """

            # Allow this inner function to update the outer maxcond variable.
            nonlocal maxcond

            try:
                # At a segment's right endpoint, use the cooling rate just
                # BEFORE the boundary, belonging to the current segment.
                #
                # np.nextafter(right, left) gives the nearest floating-point
                # number to 'right' in the direction of 'left'.
                rate_time = min(
                    t,
                    np.nextafter(right, left),
                )

                cooling_rate = float(profile.rate(rate_time))

                # physics.py calculates all eleven derivatives:
                #
                #   A @ dq/dt = b          -> nine quadrature derivatives
                #   dC/dt = -factor*dM11/dt
                #   dT/dt = -cooling_rate
                rates = state_rates(
                    z,
                    cooling_rate,
                    par,
                    condition_limit=condition_limit,
                )

                # Record the current row-scaled matrix condition number.
                #
                # state_rates() already checks this matrix. This additional
                # calculation is used to collect a diagnostic for the batch.
                A = moment_jacobian(z[:9])

                row_norms = np.linalg.norm(A, axis=1)
                A_scaled = A / row_norms[:, None]

                current_condition = float(
                    np.linalg.cond(A_scaled)
                )

                maxcond = max(
                    maxcond,
                    current_condition,
                )

                return rates

            except QuadratureFailure as error:
                # Add the solver evaluation time and profile information
                # to the original physics failure.
                #
                # This time can be a trial evaluation time, rather than
                # an accepted observation point.
                raise QuadratureFailure(
                    "Simulation stopped",
                    time_s=float(t),
                    cause=error.details,
                    profile=profile.metadata(),
                ) from error

        # ----------------------------------------------------------
        # 5. Integrate the eleven equations over this segment.
        # ----------------------------------------------------------

        sol = solve_ivp(
            fun=rhs,
            t_span=(left, right),
            y0=last,
            method=method,
            rtol=rtol,
            atol=atol,
            dense_output=True,
            max_step=min(max_step_s, right - left),
        )

        # dense_output=True creates an interpolating solution within
        # the segment. We use it below to obtain values at our chosen
        # observation times.

        if not sol.success:
            raise QuadratureFailure(
                "Integrator failed",
                time_s=float(sol.t[-1]),
                solver_message=sol.message,
            )

        # ----------------------------------------------------------
        # 6. Record the requested observations in this segment.
        # ----------------------------------------------------------

        if left == 0:
            # Include the initial state at t=0.
            mask = (times >= left) & (times <= right)
        else:
            # Exclude the left boundary because the preceding segment
            # already recorded it if it was an observation time.
            mask = (times > left) & (times <= right)

        # sol.sol(...) returns shape [11, number_of_selected_times].
        # Transpose to obtain one row per recorded time.
        segment_states = sol.sol(times[mask]).T

        states.extend(segment_states)

        # Continue the next segment from this segment's final state.
        last = sol.y[:, -1]

    # --------------------------------------------------------------
    # 7. Convert recorded states into an array and calculate moments.
    # --------------------------------------------------------------

    z = np.array(states)

    # The first nine columns contain the quadrature quantities.
    #
    # moments() calculates:
    #     M_pq = sum_i(W_i * X_i**p * Y_i**q)
    m = moments(z[:, :9])

    # --------------------------------------------------------------
    # 8. Calculate total solute mass in scaled units.
    # --------------------------------------------------------------

    # Physical mass balance:
    #
    #     dissolved concentration + crystal mass per liquid volume
    #     = c + rho * thickness * mu11
    #
    # Divide by CREF to express the total in scaled units:
    #
    #     C + factor * M11
    factor = (
        par.rho
        * par.thickness
        * NREF
        * LREF**2
        / CREF
    )

    mass = z[:, 9] + factor * m[:, INDEX[(1, 1)]]

    # --------------------------------------------------------------
    # 9. Check the recorded trajectory before returning it.
    # --------------------------------------------------------------

    # Require finite values throughout the recorded trajectory.
    # The nine quadrature variables and concentration must be positive.
    if (
        not np.isfinite(z).all()
        or np.any(z[:, :10] <= 0)
    ):
        raise QuadratureFailure("Invalid integrated states")

    # The integrated temperature should match the prescribed profile:
    #
    #     dT/dt = -cooling_rate(t)
    expected_temperature = profile.temperature(times)

    temperature_error = np.max(
        np.abs(z[:, 10] - expected_temperature)
    )

    if temperature_error > 1e-3:
        raise QuadratureFailure(
            "Temperature integration mismatch"
        )
    z[:, 10] = expected_temperature

    # np.ptp(mass) means maximum mass minus minimum mass.
    # A conserved total should have only small numerical variation.
    mass_drift = float(np.ptp(mass))

    allowed_mass_drift = (
        1e-6 * max(abs(mass[0]), 1)
    )

    if mass_drift > allowed_mass_drift:
        raise QuadratureFailure(
            "Mass conservation failed",
            drift=mass_drift,
        )

    diagnostics = {
        "max_row_scaled_condition": maxcond,
        "mass_drift_scaled": mass_drift,
    }

    return z, m, diagnostics


def run_case(case, cfg, out):
    """
    Simulate one dataset case and save its complete clean trajectory.

    case:
        Initial state and cooling-profile metadata for one batch.

    cfg:
        Configuration containing physics and solver settings.

    out:
        Destination path, normally ending in .npz.
    """

    # --------------------------------------------------------------
    # 1. Recover this case's parameters, profile, and initial state.
    # --------------------------------------------------------------

    par = Parameters(**cfg["physics"])

    # Reconstruct the exact cooling curve from its saved parameters.
    profile = from_metadata(case["profile"])

    z0 = np.array(case["initial_state11_scaled"])

    opts = cfg["solver"]
    dt = opts["sample_interval_s"]

    # Build regularly spaced observation times.
    #
    # This assumes the profile duration is an integer multiple of dt,
    # as arranged by the dataset design.
    n = int(round(profile.duration / dt))
    times = np.arange(n + 1, dtype=float) * dt

    # --------------------------------------------------------------
    # 2. Run the simulation.
    # --------------------------------------------------------------

    try:
        z, m, diagnostics = simulate(
            z0=z0,
            profile=profile,
            times=times,
            par=par,
            rtol=opts["rtol"],
            atol=opts["atol"],
            condition_limit=opts["condition_limit"],
            method=opts["method"],
            max_step_s=opts["max_step_s"],
        )

    except QuadratureFailure as error:
        # Save information about the failure for inspection.
        # Do not save the incomplete trajectory as a training example.
        path = Path(out).with_suffix(".failure.json")
        path.parent.mkdir(parents=True, exist_ok=True)

        failure_record = {
            "status": "failed",
            "case": case,
            "failure": error.details,
        }

        path.write_text(
            json.dumps(failure_record, indent=2)
        )

        # Let the caller know that this simulation failed.
        raise

    # --------------------------------------------------------------
    # 3. Prepare clean training quantities.
    # --------------------------------------------------------------

    # Each row contains:
    #     [nine scaled moments, scaled concentration]
    #
    # Temperature is saved separately.
    # No measurement noise is added in this file.
    clean = np.column_stack((m, z[:, 9]))

    # --------------------------------------------------------------
    # 4. Save the successful trajectory.
    # --------------------------------------------------------------

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,

        # Observation times, in seconds.
        t=times,

        # Full scaled eleven-variable trajectory.
        state11=z,

        # Nine scaled quadrature variables.
        q=z[:, :9],

        # Nine scaled moments plus scaled concentration.
        clean=clean,

        # Liquid temperature, in degrees Celsius.
        T=z[:, 10],

        # Moment orders, identifying the nine moment columns.
        orders=ORDERS,

        # Store settings as JSON strings inside the NPZ file.
        parameters=json.dumps(par.to_dict()),
        case=json.dumps(case),
        diagnostics=json.dumps(diagnostics),
    )

    return path


if __name__ == "__main__":
    """
    Run this file from the repository root:

        python -m crystallization.simulate

    Optional example:

        python -m crystallization.simulate \
            --config config.json \
            --case 0 \
            --out outputs/single_batch.npz
    """

    import argparse

    from .data import make_design

    # --------------------------------------------------------------
    # 1. Read command-line arguments.
    # --------------------------------------------------------------

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--config",
        default="config.json",
        help="Configuration file",
    )

    parser.add_argument(
        "--case",
        type=int,
        default=0,
        help="Index of the batch to simulate",
    )

    parser.add_argument(
        "--out",
        default="outputs/single_batch.npz",
        help="Destination for the successful trajectory",
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # 2. Load settings and construct the list of planned cases.
    # --------------------------------------------------------------

    # load_config also selects the configured moment orders.
    cfg, _ = load_config(args.config)

    # This prepares case descriptions; it does not simulate every case.
    cases = make_design(cfg)

    if not 0 <= args.case < len(cases):
        parser.error("Case index outside design")

    # --------------------------------------------------------------
    # 3. Simulate only the selected case.
    # --------------------------------------------------------------

    try:
        saved_path = run_case(
            case=cases[args.case],
            cfg=cfg,
            out=args.out,
        )

        print(saved_path)

    except QuadratureFailure as error:
        parser.exit(
            1,
            "QMOM failed; diagnostic JSON saved. "
            + str(error)
            + "\n",
        )