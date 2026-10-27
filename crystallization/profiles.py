"""Six prescribed cooling profiles for the crystallizer LIQUID.

This file defines temperature versus time. It does NOT solve the population
balance, mass balance, jacket heat transfer, or a full energy balance.

MEANING OF THE INPUTS
--------------------
start    = Ts: liquid temperature at seed addition, degrees Celsius.
end      = Tf: final liquid temperature, degrees Celsius.
duration = time taken to cool from Ts to Tf, seconds.
Th (hot preparation temperature) is not used here: this profile begins AFTER
preparation, at seed addition. Th is used elsewhere to determine initial c0.

THE TWO MAIN EQUATIONS
----------------------
s = t / duration                 normalized time: 0 at start, 1 at finish
F(s)                             fraction of total cooling completed

T(t) = Ts + (Tf - Ts) * F(s)

Differentiating with the chain rule:
dT/dt = (Tf - Ts) / duration * F'(s)

The positive cooling rate is:
cr(t) = -dT/dt = (Ts - Tf) / duration * F'(s)

Every family has F(0)=0, F(1)=1 and nonnegative F'(s). During a hold F'=0.
These are mathematical profile families, not experimentally established
actuator capabilities or a claim that they reproduce the paper's simulations.

QUICK USE
---------
profile = Profile('linear', duration=12000, start=35, end=25)
profile.temperature(6000)  # midpoint temperature, 30 degrees Celsius
profile.rate(6000)         # positive cooling rate, 10/12000 degrees C/s

Run all six examples without solving crystallization:
    python -m crystallization.profiles
or, outside the package:
    python profiles.py
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator


FAMILIES = (
    "linear",
    "cubic",
    "natural",
    "two_stage",
    "cool_hold_cool",
    "random_spline",
)


class Profile:
    """One cooling experiment, with a fixed family and fixed shape parameters.

    Random shape parameters are drawn ONCE when the object is created. Calling
    temperature() repeatedly does not redraw the curve.
    """

    def __init__(self, family, duration=10 / 0.000833,
                 start=35.0, end=25.0, rng=None):
        """Create a cooling curve.

        rng may be a NumPy random generator. Supplying a seeded generator makes
        profile creation reproducible. Without one, use a fixed seed of zero.
        """
        if family not in FAMILIES:
            raise ValueError(f"Unknown cooling family: {family}")
        if not np.isfinite([duration, start, end]).all():
            raise ValueError("Duration and temperatures must be finite")
        if duration <= 0 or start <= end or end <= 0:
            raise ValueError("Require duration > 0 and start > end > 0 Celsius")

        self.family = family
        self.duration = float(duration)
        self.start = float(start)
        self.end = float(end)

        # Create the generator only once. Reinitializing it before every draw
        # would accidentally tie unrelated shape parameters to the same draw.
        if rng is None:
            rng = np.random.default_rng(0)

        # Natural profile: larger curvature -> more front-loaded cooling.
        self.curvature = float(rng.uniform(1.0, 5.0))

        # Two-stage profile: switch is the fraction of total duration at which
        # the cooling rate changes. For example, 0.4 means t=0.4*duration.
        self.switch = float(rng.uniform(0.3, 0.7))

        # Fraction of the total temperature drop completed in the first cooling
        # stage. Used by two_stage and cool_hold_cool.
        self.drop_fraction = float(rng.uniform(0.2, 0.8))

        # Hold times are normalized fractions of duration, not temperatures.
        self.hold_start = float(rng.uniform(0.2, 0.4))
        hold_duration_fraction = float(rng.uniform(0.1, 0.3))
        self.hold_end = self.hold_start + hold_duration_fraction

        # Random spline: six normalized time coordinates, including endpoints.
        # Four internal times are sorted so they increase strictly.
        internal_times = np.sort(rng.uniform(0.05, 0.95, 4))
        self.knots = np.concatenate(([0.0], internal_times, [1.0]))

        # Positive cooling increments make the cumulative cooling monotone.
        increments = rng.uniform(0.2, 1.5, 5)
        cumulative_cooling = np.concatenate(([0.0], np.cumsum(increments)))

        # Normalize: cooling fraction starts at zero and finishes at one.
        self.values = cumulative_cooling / cumulative_cooling[-1]

        # PCHIP is a shape-preserving piecewise cubic interpolation. For these
        # increasing data it preserves monotonicity between the knots.
        self.spline = PchipInterpolator(self.knots, self.values)

    @property
    def breakpoints(self):
        """Return important segment boundaries in SECONDS.

        The simulator can integrate in separate intervals at these boundaries,
        especially where a piecewise cooling rate changes abruptly.
        Access as profile.breakpoints, without parentheses, because of @property.
        """
        if self.family == "two_stage":
            normalized_boundaries = [0.0, self.switch, 1.0]
        elif self.family == "cool_hold_cool":
            normalized_boundaries = [0.0, self.hold_start, self.hold_end, 1.0]
        elif self.family == "random_spline":
            normalized_boundaries = self.knots
        else:
            normalized_boundaries = [0.0, 1.0]

        return np.asarray(normalized_boundaries) * self.duration

    def fraction(self, s):
        """Return TWO quantities: F(s), and its derivative dF/ds.

        s is normalized time, normally in [0,1]. The derivative here is with
        respect to s, NOT seconds. rate() supplies the 1/duration conversion.
        This method handles either one s value or an array of values.
        """
        s = np.asarray(s, dtype=float)

        # FAMILY 1: LINEAR
        # F(s)=s, F'(s)=1. Equal cooling in equal time intervals.
        if self.family == "linear":
            cooling_fraction = s
            fraction_derivative = np.ones_like(s)
            return cooling_fraction, fraction_derivative

        # FAMILY 2: CUBIC
        # F(s)=s^3, F'(s)=3s^2. Slow beginning, progressively faster cooling.
        if self.family == "cubic":
            cooling_fraction = s**3
            fraction_derivative = 3 * s**2
            return cooling_fraction, fraction_derivative

        # FAMILY 3: NATURAL / EXPONENTIAL SHAPE
        # F(s)=(1-exp(-k*s))/(1-exp(-k))
        # F'(s)=k*exp(-k*s)/(1-exp(-k))
        # The denominator ensures F(1)=1 rather than an unfinished decay.
        if self.family == "natural":
            k = self.curvature
            denominator = 1.0 - np.exp(-k)

            # -expm1(-k*s) equals 1-exp(-k*s), but is more numerically accurate
            # when k*s is small.
            cooling_fraction = -np.expm1(-k * s) / denominator
            fraction_derivative = k * np.exp(-k * s) / denominator
            return cooling_fraction, fraction_derivative

        # FAMILY 4: TWO-STAGE
        # a = switching time fraction; f = first-stage cooling fraction.
        # Before a: F=f*s/a,                 F'=f/a
        # After a:  F=f+(1-f)*(s-a)/(1-a),   F'=(1-f)/(1-a)
        if self.family == "two_stage":
            a = self.switch
            f = self.drop_fraction

            first_stage = f * s / a
            second_stage = f + (1.0 - f) * (s - a) / (1.0 - a)

            # np.where chooses values element by element for an array of times.
            cooling_fraction = np.where(s < a, first_stage, second_stage)
            fraction_derivative = np.where(
                s < a,
                f / a,
                (1.0 - f) / (1.0 - a),
            )
            return cooling_fraction, fraction_derivative

        # FAMILY 5: COOL--HOLD--COOL
        # a = hold start; b = hold end; f = cooling fraction before the hold.
        # s<a:     F=f*s/a,                     F'=f/a
        # a<=s<b:  F=f,                         F'=0
        # s>=b:    F=f+(1-f)*(s-b)/(1-b),       F'=(1-f)/(1-b)
        if self.family == "cool_hold_cool":
            a = self.hold_start
            b = self.hold_end
            f = self.drop_fraction

            first_cooling = f * s / a
            second_cooling = f + (1.0 - f) * (s - b) / (1.0 - b)

            cooling_fraction = np.where(
                s < a,
                first_cooling,
                np.where(s < b, f, second_cooling),
            )
            fraction_derivative = np.where(
                s < a,
                f / a,
                np.where(s < b, 0.0, (1.0 - f) / (1.0 - b)),
            )
            return cooling_fraction, fraction_derivative

        # FAMILY 6: RANDOM MONOTONE SPLINE
        # Evaluate the fixed interpolated curve and its analytic derivative.
        # __init__ already checked the family, so this is the remaining case.
        cooling_fraction = self.spline(s)
        fraction_derivative = self.spline.derivative()(s)
        return cooling_fraction, fraction_derivative

    def temperature(self, t):
        """Return liquid temperature T(t), Celsius, for one time or many times.

        Queries outside [0,duration] are clamped to the endpoint temperatures.
        """
        normalized_time = np.clip(np.asarray(t, dtype=float) / self.duration, 0, 1)
        cooling_fraction, _ = self.fraction(normalized_time)
        return self.start + (self.end - self.start) * cooling_fraction

    def rate(self, t):
        """Return the positive cooling rate cr(t), Celsius per second.

        Formula is intended on [0,duration]. As in the original code, time is
        clamped before evaluating the slope: outside the interval this returns
        the endpoint slope, not the derivative of the clamped temperature.
        The simulator must use this function only during its cooling interval.

        At piecewise rate jumps this uses the right-hand branch; the simulator
        handles interval boundaries explicitly using breakpoints.
        """
        normalized_time = np.clip(np.asarray(t, dtype=float) / self.duration, 0, 1)
        _, fraction_derivative = self.fraction(normalized_time)

        average_cooling_rate = (self.start - self.end) / self.duration
        return average_cooling_rate * fraction_derivative

    def metadata(self):
        """Return enough information to save and exactly reconstruct this curve.

        Lists replace NumPy arrays because JSON cannot directly store arrays.
        """
        return {
            "family": self.family,
            "duration": self.duration,
            "start": self.start,
            "end": self.end,
            "knots": self.knots.tolist(),
            "values": self.values.tolist(),
            "curvature": self.curvature,
            "switch": self.switch,
            "drop_fraction": self.drop_fraction,
            "hold_start": self.hold_start,
            "hold_end": self.hold_end,
        }


def from_metadata(meta):
    """Recreate a previously saved profile without changing its shape.

    The constructor creates a temporary default shape. We overwrite all shape
    parameters with the SAVED values, then rebuild the saved spline.
    """
    profile = Profile(
        family=meta["family"],
        duration=meta["duration"],
        start=meta["start"],
        end=meta["end"],
    )

    for name in ("curvature", "switch", "drop_fraction", "hold_start", "hold_end"):
        setattr(profile, name, meta[name])

    profile.knots = np.asarray(meta["knots"], dtype=float)
    profile.values = np.asarray(meta["values"], dtype=float)
    profile.spline = PchipInterpolator(profile.knots, profile.values)
    return profile


def main():
    """Write a demonstration CSV and metadata JSON. No QMOM integration.

    Example settings below are fixed for inspecting shapes. Actual dataset
    cases elsewhere supply their own Ts, Tf, and duration.
    """
    parser = argparse.ArgumentParser(description="Write all six cooling profiles")
    parser.add_argument("--out", default="outputs/profiles.csv")
    args = parser.parse_args()

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration = 12000.0  # seconds = 200 minutes
    seeding_temperature = 35.0
    final_temperature = 25.0

    profiles = []
    for index, family in enumerate(FAMILIES):
        # Different seed for each family, reproducible between executions.
        generator = np.random.default_rng(42 + index)
        profiles.append(Profile(
            family=family,
            duration=duration,
            start=seeding_temperature,
            end=final_temperature,
            rng=generator,
        ))

    # This CSV records 121 equally spaced times: 0,100,...,12000 seconds.
    # It is only a demonstration grid, not the dataset observation interval.
    times = np.linspace(0.0, duration, 121)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time_s"] + [profile.family + "_C" for profile in profiles])
        for time in times:
            writer.writerow([time] + [float(profile.temperature(time)) for profile in profiles])

    # Save exact random parameters so curves can be rebuilt later.
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps([profile.metadata() for profile in profiles], indent=2),
        encoding="utf-8",
    )

    print(f"Temperature curves: {output_path}")
    print(f"Profile parameters: {metadata_path}")


if __name__ == "__main__":
    main()
