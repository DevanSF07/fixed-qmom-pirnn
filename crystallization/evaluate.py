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

Both forecasting scopes are reported separately under distinct metric sections.
One scope does not validate the other.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

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
    Parameters,
    set_orders,
    torch_moment_rates,
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def load_model(checkpoint):
    """
    Load model from checkpoint with strict state_dict verification.
    No silent omission or tolerance for mismatched parameter keys.
    """
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = ck["config"]
    set_orders(cfg["moments"])
    par = Parameters(**cfg["physics"])
    model = Forecaster(cfg["training"]["hidden"], par=par)
    # STRICT loading: ensures all keys match exactly and none are missing or unexpected
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, cfg, ck


def predict(checkpoint, history, temperature):
    """
    Predict next horizon steps given history [L, 11] and future temperature [H+1].
    """
    model, cfg, _ = load_model(checkpoint)
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
    rollouts=4,
):
    torch.set_num_threads(4)
    model, cfg, ck = load_model(checkpoint)
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
    checkpoint_hash = sha256_file(checkpoint)
    manifest_path = Path(data) / "manifest.json"
    manifest_hash = sha256_file(manifest_path)
    git_commit = get_git_commit()

    scales = np.r_[[NREF * LREF ** (p + q) for p, q in ORDERS], CREF]
    factor = ds.par.rho * ds.par.thickness * NREF * LREF**2 / CREF

    # -------------------------------------------------------------
    # 1. Short-Horizon Window Evaluation (Observation Refreshed)
    # -------------------------------------------------------------
    squared = np.zeros(10)
    abs_err = np.zeros(10)
    all_truth = []
    length_errors = []
    width_errors = []
    own_mass_drifts = []
    true_mass_errors = []
    neg_c_count = 0
    neg_c_min = 0.0
    q_bound_hits = 0
    total_steps = 0
    res_norms = []

    with torch.no_grad():
        for h, T, y in DataLoader(ds, batch_size=s["batch_size"]):
            res = model(h, T)
            pred = res["z"][:, 1:]
            q_pred = res["q"][:, 1:]

            if not torch.isfinite(pred).all():
                raise RuntimeError("Nonfinite predictions in short-horizon windows")

            diff = (pred - y[:, 1:]).double()
            squared += (diff.square().sum((0, 1))).numpy()
            abs_err += (diff.abs().sum((0, 1))).numpy()
            all_truth.append(y[:, 1:].numpy())

            # Mean crystal length & width errors
            # Length: mu10 / mu00; Width: mu01 / mu00
            l_truth = (y[:, 1:, INDEX[(1, 0)]] / y[:, 1:, 0]) * LREF * 1e6
            l_pred = (pred[:, :, INDEX[(1, 0)]] / pred[:, :, 0]) * LREF * 1e6
            length_errors.append((l_pred - l_truth).abs().numpy())

            w_truth = (y[:, 1:, INDEX[(0, 1)]] / y[:, 1:, 0]) * LREF * 1e6
            w_pred = (pred[:, :, INDEX[(0, 1)]] / pred[:, :, 0]) * LREF * 1e6
            width_errors.append((w_pred - w_truth).abs().numpy())

            # Mass conservation checks:
            # Model's own mass: C(t) + factor * M11(t)
            # True initial mass: C_truth(0) + factor * M11_truth(0)
            m_pred_mass = pred[:, :, 9] + factor * pred[:, :, INDEX[(1, 1)]]
            m_own_drift = (m_pred_mass - m_pred_mass[:, :1]).abs().max(dim=1).values
            own_mass_drifts.extend(m_own_drift.numpy().tolist())

            m_true_init = y[:, :1, 9] + factor * y[:, :1, INDEX[(1, 1)]]
            m_true_err = (m_pred_mass - m_true_init).abs().max(dim=1).values
            true_mass_errors.extend(m_true_err.numpy().tolist())

            # Constraint activations: negative concentration
            c_pred = pred[:, :, 9].numpy()
            neg_mask = c_pred < 0
            neg_c_count += int(np.sum(neg_mask))
            if np.any(neg_mask):
                neg_c_min = min(neg_c_min, float(np.min(c_pred)))

            # Quadrature bounds activations: q in [1e-6, 25.0]
            q_np = q_pred.numpy()
            q_hits = np.isclose(q_np, 1e-6, atol=1e-7) | np.isclose(q_np, 25.0, atol=1e-5)
            q_bound_hits += int(np.sum(q_hits))

            # ODE moment rate residual norm
            dm = torch_moment_rates(res["q"], res["z"][..., 9], T, ds.par)
            f = torch.cat((dm, -factor * dm[..., INDEX[(1, 1)] : INDEX[(1, 1)] + 1]), -1)
            residual = res["z"][:, 1:] - res["z"][:, :-1] - 0.5 * ds.dt * (f[:, 1:] + f[:, :-1])
            res_norms.extend(torch.norm(residual, dim=-1).mean(dim=1).numpy().tolist())

            total_steps += pred.shape[0] * pred.shape[1]

    n = total_steps
    rmse = np.sqrt(squared / n)
    mae = abs_err / n

    # Compute normalized errors: denominator = truth range (max - min)
    truth_all_cat = np.concatenate(all_truth, axis=0).reshape(-1, 10)
    truth_ranges = np.ptp(truth_all_cat, axis=0)
    denom_norm = np.maximum(truth_ranges, 1e-12)
    norm_rmse = rmse / denom_norm
    norm_mae = mae / denom_norm

    l_err_cat = np.concatenate(length_errors, axis=0)
    w_err_cat = np.concatenate(width_errors, axis=0)

    manifest = json.loads(manifest_path.read_text())

    metrics = {
        "metadata": {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "data_manifest": str(manifest_path),
            "data_manifest_sha256": manifest_hash,
            "git_commit": git_commit,
            "architecture": ck.get("architecture", {}),
            "units": {
                "concentration": "kg/m^3 (physical = scaled * CREF, CREF = 240.0 kg/m^3)",
                "moments": "physical mu_pq = scaled M_pq * NREF * LREF^(p+q)",
                "lengths": "micrometres (um)",
                "normalized_errors_denominator": "Range of ground truth in test set: max(truth) - min(truth)",
            },
        },
        "short_horizon_windows": {
            "description": "10-step (100s) forecast horizon, conditioned on latest 5 observed steps",
            "test_windows_count": len(ds),
            "test_batches_count": len(ds.batches),
            "scaled_rmse": dict(zip(NAMES + ["c"], rmse.tolist())),
            "normalized_rmse_pct": dict(zip(NAMES + ["c"], (norm_rmse * 100.0).tolist())),
            "physical_rmse": dict(zip(NAMES + ["c_kg_m3"], (rmse * scales).tolist())),
            "scaled_mae": dict(zip(NAMES + ["c"], mae.tolist())),
            "normalized_mae_pct": dict(zip(NAMES + ["c"], (norm_mae * 100.0).tolist())),
            "physical_mae": dict(zip(NAMES + ["c_kg_m3"], (mae * scales).tolist())),
            "mean_crystal_length_mae_um": float(np.mean(l_err_cat)),
            "mean_crystal_width_mae_um": float(np.mean(w_err_cat)),
            "mass_errors": {
                "model_own_mass_drift_max_scaled": float(np.max(own_mass_drifts)),
                "model_own_mass_drift_mean_scaled": float(np.mean(own_mass_drifts)),
                "error_rel_true_initial_mass_max_scaled": float(np.max(true_mass_errors)),
                "error_rel_true_initial_mass_mean_scaled": float(np.mean(true_mass_errors)),
            },
            "constraint_activations": {
                "negative_concentration_steps": neg_c_count,
                "negative_concentration_rate": float(neg_c_count / n),
                "negative_concentration_min_scaled": float(neg_c_min),
                "negative_concentration_min_kg_m3": float(neg_c_min * CREF),
                "quadrature_bound_hits": q_bound_hits,
            },
            "mean_physics_residual_norm": float(np.mean(res_norms)),
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
            "an initial 50s seed window. Results for both scopes are reported independently. "
            "Finite status indicates numerical non-divergence; physical accuracy is reported separately."
        ),
    }

    # -------------------------------------------------------------
    # 2. Autonomous Full-Batch Rollout Evaluation
    # -------------------------------------------------------------
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
            q_out = full_pred["q"][0, 1:].numpy()
            truth = batch["clean"][L:]

            if not np.isfinite(z).all():
                metrics["autonomous_rollouts"].append(
                    {
                        "id": batch["id"],
                        "status": "nonfinite",
                        "steps_completed": 0,
                    }
                )
                continue

            # Model's own mass drift vs error relative to true initial mass
            mass_pred = z[:, 9] + factor * z[:, INDEX[(1, 1)]]
            m0_model = mass_pred[0]
            model_own_drift = float(np.max(np.abs(mass_pred - m0_model)))

            m0_truth = batch["clean"][L - 1, 9] + factor * batch["clean"][L - 1, INDEX[(1, 1)]]
            err_rel_true = float(np.max(np.abs(mass_pred - m0_truth)))

            scaled_rmse = np.sqrt(np.mean((z - truth) ** 2, axis=0))
            scaled_mae = np.mean(np.abs(z - truth), axis=0)
            physical_rmse = scaled_rmse * scales
            physical_mae = scaled_mae * scales

            truth_ptp = np.ptp(truth, axis=0)
            norm_rollout_rmse = scaled_rmse / np.maximum(truth_ptp, 1e-12)
            norm_rollout_mae = scaled_mae / np.maximum(truth_ptp, 1e-12)

            length_truth_um = (truth[:, INDEX[(1, 0)]] / truth[:, 0]) * LREF * 1e6
            length_pred_um = (z[:, INDEX[(1, 0)]] / z[:, 0]) * LREF * 1e6
            length_rmse_um = float(np.sqrt(np.mean((length_pred_um - length_truth_um) ** 2)))
            length_mae_um = float(np.mean(np.abs(length_pred_um - length_truth_um)))

            width_truth_um = (truth[:, INDEX[(0, 1)]] / truth[:, 0]) * LREF * 1e6
            width_pred_um = (z[:, INDEX[(0, 1)]] / z[:, 0]) * LREF * 1e6
            width_rmse_um = float(np.sqrt(np.mean((width_pred_um - width_truth_um) ** 2)))
            width_mae_um = float(np.mean(np.abs(width_pred_um - width_truth_um)))

            # Constraint activations in autonomous rollout
            c_rollout = z[:, 9]
            c_neg_count = int(np.sum(c_rollout < 0))
            c_min_val = float(np.min(c_rollout))
            q_bounds_count = int(np.sum(np.isclose(q_out, 1e-6, atol=1e-7) | np.isclose(q_out, 25.0, atol=1e-5)))

            # Physics residual norm along autonomous rollout
            T_tensor = torch.tensor(batch["T"][L:][None], dtype=torch.float32)
            z_tensor = torch.tensor(z[None], dtype=torch.float32)
            q_tensor = torch.tensor(q_out[None], dtype=torch.float32)
            dm_rollout = torch_moment_rates(q_tensor, z_tensor[..., 9], T_tensor, ds.par)
            f_rollout = torch.cat((dm_rollout, -factor * dm_rollout[..., INDEX[(1, 1)] : INDEX[(1, 1)] + 1]), -1)
            res_rollout = z_tensor[:, 1:] - z_tensor[:, :-1] - 0.5 * ds.dt * (f_rollout[:, 1:] + f_rollout[:, :-1])
            mean_res_norm = float(torch.norm(res_rollout, dim=-1).mean().item())

            rollout_record = {
                "id": batch["id"],
                "status": "finite",
                "rollout_steps": int(z.shape[0]),
                "rollout_duration_s": float(batch["t"][-1] - batch["t"][L]),
                "mass_conservation": {
                    "model_own_mass_drift_scaled": model_own_drift,
                    "error_rel_true_initial_mass_scaled": err_rel_true,
                },
                "constraint_activations": {
                    "negative_concentration_steps": c_neg_count,
                    "min_concentration_scaled": c_min_val,
                    "min_concentration_kg_m3": c_min_val * CREF,
                    "quadrature_bound_hits": q_bounds_count,
                },
                "physics_residual_norm": mean_res_norm,
                "mean_crystal_length_rmse_um": length_rmse_um,
                "mean_crystal_length_mae_um": length_mae_um,
                "mean_crystal_width_rmse_um": width_rmse_um,
                "mean_crystal_width_mae_um": width_mae_um,
                "scaled_rmse": dict(zip(NAMES + ["c"], scaled_rmse.tolist())),
                "normalized_rmse_pct": dict(zip(NAMES + ["c"], (norm_rollout_rmse * 100.0).tolist())),
                "physical_rmse": dict(zip(NAMES + ["c_kg_m3"], physical_rmse.tolist())),
                "scaled_mae": dict(zip(NAMES + ["c"], scaled_mae.tolist())),
                "normalized_mae_pct": dict(zip(NAMES + ["c"], (norm_rollout_mae * 100.0).tolist())),
                "physical_mae": dict(zip(NAMES + ["c_kg_m3"], physical_mae.tolist())),
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
    parser.add_argument("--rollouts", type=int, default=4, help="Number of full-batch rollouts to evaluate")
    args = parser.parse_args()

    results = evaluate(**vars(args))
    print(json.dumps(results, indent=2))
