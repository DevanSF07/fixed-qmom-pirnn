"""
Inspect PIRNN predictions on held-out test data.

Evaluation Scopes:
1. Short-Horizon Forecasts (Windows):
   Refreshed with observations at each window (history = 5 steps, horizon = 10 steps = 100s).
   Evaluates local one-step/short-horizon predictive accuracy.

2. Autonomous Full-Batch Rollouts:
   Initiated from the initial history window (t <= 50s) and rolled out unassisted over the
   entire batch duration (~6,000s) driven only by the prescribed liquid temperature curve.
   Evaluates long-term dynamical stability, accumulated drift, and mass conservation.

These two forecasting scopes are reported separately under distinct metric sections.
One scope does not validate the other.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import Windows
from .model import Forecaster
from .physics import (
    CREF,
    INDEX,
    LREF,
    NAMES,
    NREF,
    ORDERS,
    set_orders,
)


def load_model(checkpoint):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = ck["config"]
    set_orders(cfg["moments"])
    from .physics import Parameters
    par = Parameters(**cfg["physics"])
    model = Forecaster(cfg["training"]["hidden"], par=par)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg


def predict(checkpoint, history, temperature):
    """
    Predict next horizon steps given history [L, 11] and future temperature [H+1].
    """
    model, cfg = load_model(checkpoint)
    history = np.asarray(history, dtype=np.float32)
    temperature = np.asarray(temperature, dtype=np.float32)
    if (
        history.shape != (cfg["training"]["history"], 11)
        or temperature.ndim != 1
        or len(temperature) < 2
    ):
        raise ValueError("Input shapes differ from checkpoint configuration")

    with torch.no_grad():
        result = model(
            torch.tensor(history[None], dtype=torch.float32),
            torch.tensor(temperature[None], dtype=torch.float32),
        )
    return {k: v[0].numpy() for k, v in result.items()}


def evaluate(
    data="outputs/data",
    checkpoint="outputs/pirnn/best.pt",
    out="outputs/evaluation",
    rollouts=2,
):
    torch.set_num_threads(4)
    model, cfg = load_model(checkpoint)
    s = cfg["training"]
    ds = Windows(data, "test", s["history"], s["horizon"])

    if (
        cfg["physics"] != ds.cfg["physics"]
        or cfg["moments"] != ds.cfg["moments"]
        or not np.isclose(ds.dt, cfg["solver"]["sample_interval_s"])
    ):
        raise ValueError("Checkpoint physics or timestep mismatch with dataset")

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Short-Horizon Window Evaluation (Observation Refreshed)
    # -------------------------------------------------------------
    squared = np.zeros(10)
    abs_err = np.zeros(10)
    n = 0

    with torch.no_grad():
        for h, T, y in DataLoader(ds, batch_size=s["batch_size"]):
            pred = model(h, T)["z"][:, 1:]
            if not torch.isfinite(pred).all():
                raise RuntimeError("Nonfinite predictions in short-horizon windows")
            diff = (pred - y[:, 1:]).double()
            squared += (diff.square().sum((0, 1))).numpy()
            abs_err += (diff.abs().sum((0, 1))).numpy()
            n += pred.shape[0] * pred.shape[1]

    rmse = np.sqrt(squared / n)
    mae = abs_err / n
    scales = np.r_[[NREF * LREF ** (p + q) for p, q in ORDERS], CREF]

    manifest = json.loads((Path(data) / "manifest.json").read_text())

    metrics = {
        "short_horizon_windows": {
            "description": "10-step (100s) forecast horizon, conditioned on latest 5 observed steps",
            "test_windows_count": len(ds),
            "test_batches_count": len(ds.batches),
            "scaled_rmse": dict(zip(NAMES + ["c"], rmse.tolist())),
            "physical_rmse": dict(zip(NAMES + ["c"], (rmse * scales).tolist())),
            "scaled_mae": dict(zip(NAMES + ["c"], mae.tolist())),
            "physical_mae": dict(zip(NAMES + ["c"], (mae * scales).tolist())),
        },
        "autonomous_rollouts": [],
        "dataset_summary": {
            "completed_batches": len(manifest["completed"]),
            "failed_batches": len(manifest["failed"]),
            "total_planned": manifest["planned"],
        },
        "notes": (
            "Short-horizon window evaluation refreshes the hidden state using recent observations. "
            "Autonomous rollout evaluates open-loop integration over the full batch (~6,000s) from "
            "an initial 50s seed window. Results for both scopes are reported independently."
        ),
    }

    # -------------------------------------------------------------
    # 2. Autonomous Full-Batch Rollout Evaluation
    # -------------------------------------------------------------
    factor = ds.par.rho * ds.par.thickness * NREF * LREF**2 / CREF
    L = s["history"]

    with torch.no_grad():
        for batch in ds.batches[: max(0, rollouts)]:
            obs = batch["observed"][:L]
            h = np.column_stack((np.log1p(obs[:, :10]), (obs[:, 10] - 30.0) / 5.0))

            full_pred = model(
                torch.tensor(h[None], dtype=torch.float32),
                torch.tensor(batch["T"][L - 1 :][None], dtype=torch.float32),
            )
            z = full_pred["z"][0, 1:].numpy()
            truth = batch["clean"][L:]

            if not np.isfinite(z).all():
                metrics["autonomous_rollouts"].append(
                    {
                        "id": batch["id"],
                        "status": "NONFINITE_PREDICTION",
                        "steps_completed": 0,
                    }
                )
                continue

            mass_pred = z[:, 9] + factor * z[:, INDEX[(1, 1)]]
            m0_truth = batch["clean"][L - 1, 9] + factor * batch["clean"][L - 1, INDEX[(1, 1)]]
            mass_drift = float(np.max(np.abs(mass_pred - m0_truth)))

            scaled_rmse = np.sqrt(np.mean((z - truth) ** 2, axis=0))
            physical_rmse = scaled_rmse * scales

            length_truth_um = (truth[:, INDEX[(1, 0)]] / truth[:, 0]) * LREF * 1e6
            length_pred_um = (z[:, INDEX[(1, 0)]] / z[:, 0]) * LREF * 1e6
            length_rmse_um = float(np.sqrt(np.mean((length_pred_um - length_truth_um) ** 2)))

            width_truth_um = (truth[:, INDEX[(0, 1)]] / truth[:, 0]) * LREF * 1e6
            width_pred_um = (z[:, INDEX[(0, 1)]] / z[:, 0]) * LREF * 1e6
            width_rmse_um = float(np.sqrt(np.mean((width_pred_um - width_truth_um) ** 2)))

            rollout_record = {
                "id": batch["id"],
                "status": "FINITE_SUCCESS",
                "rollout_steps": int(z.shape[0]),
                "rollout_duration_s": float(batch["t"][-1] - batch["t"][L]),
                "max_mass_error_scaled": mass_drift,
                "mean_crystal_length_rmse_um": length_rmse_um,
                "mean_crystal_width_rmse_um": width_rmse_um,
                "scaled_rmse": dict(zip(NAMES + ["c"], scaled_rmse.tolist())),
                "physical_rmse": dict(zip(NAMES + ["c"], physical_rmse.tolist())),
            }
            metrics["autonomous_rollouts"].append(rollout_record)

            np.savez_compressed(
                out / f"{batch['id']}.npz",
                prediction=z,
                truth=truth,
                t=batch["t"][L:],
            )

            # Generate diagnostic comparison plots
            fig, axes = plt.subplots(5, 2, figsize=(14, 16))
            t_plot = batch["t"][L:]
            for j, ax in enumerate(axes.flat):
                var_name = (NAMES + ["c"])[j]
                ax.plot(t_plot, truth[:, j], label="Clean Simulator", color="black", lw=1.5)
                ax.plot(t_plot, z[:, j], label="PIRNN Rollout", color="crimson", ls="--", lw=1.5)
                ax.set_title(f"{var_name} (scaled)", fontsize=11, fontweight="bold")
                ax.set_xlabel("Time [s]")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="best")

            fig.suptitle(
                f"Autonomous Full-Batch PIRNN Rollout vs Simulator: {batch['id']}",
                fontsize=13,
                fontweight="bold",
                y=0.995,
            )
            fig.tight_layout()
            fig.savefig(out / f"{batch['id']}.png", dpi=130)
            plt.close(fig)

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="outputs/data", help="Data folder containing manifest.json")
    parser.add_argument("--checkpoint", default="outputs/pirnn/best.pt", help="Trained model checkpoint")
    parser.add_argument("--out", default="outputs/evaluation", help="Destination folder for metrics and plots")
    parser.add_argument("--rollouts", type=int, default=2, help="Number of full-batch rollouts to evaluate")
    args = parser.parse_args()

    results = evaluate(**vars(args))
    print(json.dumps(results, indent=2))
