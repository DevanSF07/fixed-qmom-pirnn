"""
Generate the exact deliverables requested:
1. Graph of the generated data: clean (before noise) vs. observed (with added noise).
2. Validation results:
   A. Loss curves (training components and validation loss across epochs).
   B. Loss and error metrics for each predicted feature (9 moments + concentration).
3. Clean up extraneous output files.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from crystallization.data import Windows
from crystallization.model import Forecaster
from crystallization.physics import (
    CREF,
    LREF,
    NAMES,
    NREF,
    ORDERS,
    Parameters,
    load_config,
    set_orders,
)

def generate_clean_vs_noisy_plot(data_path, out_dir):
    data = np.load(data_path, allow_pickle=True)
    t = data["t"]
    clean = data["clean"]        # [N, 10]: 9 moments + concentration
    observed = data["observed"]  # [N, 11]: 9 moments + concentration + temperature
    T_clean = data["T"]          # [N]
    
    # 11 variables: 9 moments, concentration, temperature
    fig, axes = plt.subplots(4, 3, figsize=(16, 14), dpi=200)
    axes = axes.flatten()
    
    features = NAMES + ["c (Concentration)", "T (Liquid Temperature)"]
    units = [
        r"$\mathrm{scaled}$" for _ in NAMES
    ] + [r"$\mathrm{scaled}\ (c/240)$", r"$^\circ\mathrm{C}$"]
    
    # Plot first 9 moments
    for i, name in enumerate(NAMES):
        ax = axes[i]
        ax.plot(t, clean[:, i], label="Before Noise (Clean Ground Truth)", color="#1f77b4", lw=2.0)
        ax.plot(t, observed[:, i], label="With Added Noise (Observed)", color="#d62728", ls="--", lw=1.2, alpha=0.75)
        ax.set_title(f"{name} ({units[i]})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time [s]", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)
        if i == 0:
            ax.legend(fontsize=8, loc="best", framealpha=0.9)
            
    # Concentration (index 9)
    ax_c = axes[9]
    ax_c.plot(t, clean[:, 9], label="Before Noise (Clean Ground Truth)", color="#1f77b4", lw=2.0)
    ax_c.plot(t, observed[:, 9], label="With Added Noise (Observed)", color="#d62728", ls="--", lw=1.2, alpha=0.75)
    ax_c.set_title(f"Concentration c ({units[9]})", fontsize=11, fontweight="bold")
    ax_c.set_xlabel("Time [s]", fontsize=9)
    ax_c.grid(True, linestyle=":", alpha=0.6)
    
    # Temperature (clean is T_clean, observed is observed[:, 10])
    ax_T = axes[10]
    ax_T.plot(t, T_clean, label="Before Noise (Prescribed Profile)", color="#1f77b4", lw=2.0)
    ax_T.plot(t, observed[:, 10], label="With Added Noise (Sensor Obs)", color="#d62728", ls="--", lw=1.2, alpha=0.75)
    ax_T.set_title(f"Temperature T ({units[10]})", fontsize=11, fontweight="bold")
    ax_T.set_xlabel("Time [s]", fontsize=9)
    ax_T.grid(True, linestyle=":", alpha=0.6)
    
    # 12th subplot: Noise comparison summary box
    ax_info = axes[11]
    ax_info.axis("off")
    noise_cfg = {
        "Temperature Std Dev": "0.1 °C",
        "Concentration Std Dev": "0.3 kg/m³ (scaled: ~0.00125)",
        "Node Weight CV": "5%",
        "Node Length CV": "2%",
        "Node Width CV": "2%"
    }
    info_text = "Synthetic Noise Specifications:\n" + "-"*35 + "\n"
    for k, v in noise_cfg.items():
        info_text += f"• {k}: {v}\n"
    info_text += "\nLegend:\n— Solid Blue: Clean Physical Simulation\n-- Dashed Red: Observation with Sensor Noise"
    ax_info.text(0.1, 0.5, info_text, transform=ax_info.transAxes, fontsize=10,
                 verticalalignment="center", bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor="#ced4da"))
    
    fig.suptitle("Generated Data Comparison: Before Noise (Clean) vs. With Added Noise (Observed)\nBatch: g001_s00_linear",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    out_file = out_dir / "1_generated_data_clean_vs_noisy.png"
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_file}")


def generate_loss_curve_plot(training_json_path, out_dir):
    history = json.loads(Path(training_json_path).read_text())
    epochs = [h["epoch"] for h in history]
    data_loss = [h["data"] for h in history]
    shape_loss = [h.get("shape", 0.0) for h in history]
    physics_loss = [h["physics"] for h in history]
    mass_loss = [h["mass"] for h in history]
    val_loss = [h.get("val_loss", h.get("val_log_mse")) for h in history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=200)
    
    # Left: Training Loss Components
    ax1.plot(epochs, data_loss, marker="o", markersize=4, label="Data Loss (Weighted Log MSE)", color="#1f77b4", lw=2.0)
    if any(s > 0 for s in shape_loss):
        ax1.plot(epochs, shape_loss, marker="^", markersize=4, label="Shape Ratio Loss (Mean Length & Width)", color="#e377c2", lw=2.0)
    ax1.plot(epochs, physics_loss, marker="s", markersize=4, label="Physics Loss (ODE Residual)", color="#2ca02c", lw=2.0)
    ax1.plot(epochs, mass_loss, marker="v", markersize=4, label="Mass Conservation Loss", color="#ff7f0e", lw=2.0)
    ax1.set_title("Training Loss Components vs. Epoch", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss Magnitude (Log Scale)", fontsize=11)
    ax1.set_yscale("log")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=9, loc="best", framealpha=0.9)
    
    # Right: Validation Loss Curve
    best_epoch = int(np.argmin(val_loss)) + 1
    best_val = min(val_loss)
    
    ax2.plot(epochs, val_loss, marker="D", markersize=4, label="Validation Objective", color="#9467bd", lw=2.2)
    ax2.scatter([best_epoch], [best_val], color="#d62728", s=120, zorder=5, label=f"Best Checkpoint (Epoch {best_epoch}, {best_val:.6f})")
    ax2.axvline(best_epoch, color="#d62728", linestyle=":", alpha=0.7)
    ax2.set_title("Validation Loss Curve vs. Epoch", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Validation Loss", fontsize=11)
    ax2.set_yscale("log")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=10, loc="best", framealpha=0.9)
    
    fig.suptitle("PIRNN Training & Validation Convergence Over 30 Epochs", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    out_file = out_dir / "2_training_validation_loss_curves.png"
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_file}")


def evaluate_feature_losses(data_dir, checkpoint_path, out_dir):
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = ck["config"]
    set_orders(cfg["moments"])
    par = Parameters(**cfg["physics"])
    model = Forecaster(cfg["training"]["hidden"], par=par)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    
    s = cfg["training"]
    # Evaluate on VALIDATION split (Group 9)
    val_ds = Windows(data_dir, "val", s["history"], s["horizon"])
    
    squared = np.zeros(10)
    abs_err = np.zeros(10)
    all_truth = []
    total_steps = 0
    
    scales = np.r_[[NREF * LREF ** (p + q) for p, q in ORDERS], CREF]
    feature_names = NAMES + ["c"]
    
    with torch.no_grad():
        for h, T, y in DataLoader(val_ds, batch_size=s["batch_size"]):
            pred = model(h, T)["z"][:, 1:]
            diff = (pred - y[:, 1:]).double()
            squared += (diff.square().sum((0, 1))).numpy()
            abs_err += (diff.abs().sum((0, 1))).numpy()
            all_truth.append(y[:, 1:].numpy())
            total_steps += pred.shape[0] * pred.shape[1]
            
    mse = squared / total_steps
    rmse = np.sqrt(mse)
    mae = abs_err / total_steps
    
    truth_cat = np.concatenate(all_truth, axis=0).reshape(-1, 10)
    truth_ptp = np.ptp(truth_cat, axis=0)
    norm_rmse_pct = (rmse / np.maximum(truth_ptp, 1e-12)) * 100.0
    norm_mae_pct = (mae / np.maximum(truth_ptp, 1e-12)) * 100.0
    physical_rmse = rmse * scales
    physical_mae = mae * scales
    
    results = {
        "validation_windows_count": len(val_ds),
        "validation_steps_count": total_steps,
        "features": {}
    }
    
    for i, name in enumerate(feature_names):
        results["features"][name] = {
            "scaled_mse": float(mse[i]),
            "scaled_rmse": float(rmse[i]),
            "scaled_mae": float(mae[i]),
            "normalized_rmse_pct": float(norm_rmse_pct[i]),
            "normalized_mae_pct": float(norm_mae_pct[i]),
            "physical_rmse": float(physical_rmse[i]),
            "physical_mae": float(physical_mae[i]),
            "physical_unit": "kg/m^3" if name == "c" else f"m^({ORDERS[i][0]+ORDERS[i][1]}-3)"
        }
        
    (out_dir / "validation_feature_losses.json").write_text(json.dumps(results, indent=2))
    print(f"Saved: {out_dir / 'validation_feature_losses.json'}")
    
    # Plot feature loss / error
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=200)
    
    # Bar Chart 1: Normalized RMSE & MAE (%)
    x = np.arange(len(feature_names))
    width = 0.38
    rects1 = ax1.bar(x - width/2, norm_rmse_pct, width, label="Normalized RMSE (%)", color="#1f77b4", alpha=0.85)
    rects2 = ax1.bar(x + width/2, norm_mae_pct, width, label="Normalized MAE (%)", color="#ff7f0e", alpha=0.85)
    
    ax1.set_title("Validation Prediction Error per Feature (Normalized %)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(feature_names, fontsize=10, rotation=25)
    ax1.set_ylabel("Error Normalized by Ground Truth Range [%]", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax1.legend(fontsize=10, loc="upper right")
    
    # Add value labels above bars
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)

    # Bar Chart 2: Scaled MSE (Log Scale)
    rects3 = ax2.bar(x, mse, width=0.55, color="#2ca02c", alpha=0.85, label="Feature Mean Squared Error (MSE)")
    ax2.set_title("Validation Mean Squared Error (MSE) per Feature", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(feature_names, fontsize=10, rotation=25)
    ax2.set_ylabel("MSE (Scaled, Log Scale)", fontsize=10)
    ax2.set_yscale("log")
    ax2.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax2.legend(fontsize=10, loc="upper right")
    
    for rect in rects3:
        h = rect.get_height()
        ax2.annotate(f"{h:.1e}", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)

    fig.suptitle("Validation Performance Across All 10 Predicted Features (9 Moments + Concentration)",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    out_file = out_dir / "3_loss_per_feature.png"
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_file}")
    
    return results

def cleanup_others(root_dir):
    """Remove extraneous diagnostic folders and unneeded temporary files."""
    import shutil
    root = Path(root_dir)
    to_remove_dirs = [
        root / "outputs" / "data_growth_only_diagnostic",
    ]
    to_remove_files = [
        root / "outputs" / "single_batch.npz",
        root / "outputs" / "evaluation" / "g001_s00_cubic.npz",
        root / "outputs" / "evaluation" / "g001_s00_linear.npz",
        root / "outputs" / "evaluation" / "g001_s00_natural.npz",
        root / "outputs" / "evaluation" / "g001_s00_two_stage.npz",
        root / "outputs" / "evaluation" / "g001_s00_cubic.png",
        root / "outputs" / "evaluation" / "g001_s00_linear.png",
        root / "outputs" / "evaluation" / "g001_s00_natural.png",
        root / "outputs" / "evaluation" / "g001_s00_two_stage.png",
    ]
    for d in to_remove_dirs:
        if d.exists() and d.is_dir():
            shutil.rmtree(d)
            print(f"Removed directory: {d}")
    for f in to_remove_files:
        if f.exists() and f.is_file():
            f.unlink()
            print(f"Removed file: {f}")

def main():
    root = Path(".")
    out_dir = root / "outputs" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Clean vs Noisy plot
    sample_file = root / "outputs" / "data" / "g001_s00_linear.npz"
    generate_clean_vs_noisy_plot(sample_file, out_dir)
    
    # 2. Loss curve plot
    training_json = root / "outputs" / "pirnn" / "training.json"
    generate_loss_curve_plot(training_json, out_dir)
    
    # 3. Feature loss evaluation & plot
    checkpoint = root / "outputs" / "pirnn" / "best.pt"
    evaluate_feature_losses(root / "outputs" / "data", checkpoint, out_dir)
    
    # 4. Clean up other clutter
    cleanup_others(root)
    print("\nAll deliverables generated and extraneous files removed successfully!")

if __name__ == "__main__":
    main()
