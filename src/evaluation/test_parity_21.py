"""
plot_forces_parity_all.py
-------------------------
Evaluates a trained E3NN force model over ALL rows in index_test.csv and
produces:
  1) An aggregate parity plot: NN force (y) vs DFT force (x) for *all* samples.
  2) (Optional) Per-sample parity plots saved to a folder.

Edit the CONFIG block below.
"""

import os, sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# USER CONFIG
# ---------------------------------------------------------------------
CONFIG = dict(
    model_path        = r"C:\Users\labadmin\Documents\GNN imp\src\models\latest_checkpoint.pt", # <-- IMPORTANT: UPDATE THIS PATH
    index_csv         = r"C:\Users\labadmin\Documents\GNN imp\src\data\index_eq.csv", 
    root_dir          = r"D:/Sara/All_DFT_Data",
    out_dir           = "parity_plots",         # folder to write outputs
    aggregate_png     = "forces_parity_ALL.png",
    dpi               = 300,
    make_per_sample   = False,                  # True to save per-sample plots as well
    per_sample_prefix = "forces_parity_",       # file name prefix for per-sample plots
    strict_load       = True,                   # set False only if you *intentionally* changed the model
)
# ---------------------------------------------------------------------

# If this file lives inside /Code/ML/, make parent visible in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.e3nn_gnn_model import E3NNForceModel, compute_forces
from data.lazy_graph_dataset import LazyGraphDataset

def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def forces_parity_plot(x_true: np.ndarray, y_pred: np.ndarray,
                       title: str,
                       out_png: str,
                       dpi: int = 200,
                       show: bool = False):
    """Generic parity plot helper. x_true and y_pred are 1D arrays (flattened forces)."""
    mae  = float(np.mean(np.abs(y_pred - x_true)))
    rmse = float(np.sqrt(np.mean((y_pred - x_true) ** 2)))
    r2   = float(_r2_score(x_true, y_pred))

    lim = np.max(np.abs(np.concatenate([x_true, y_pred]))) * 1.05
    lim = max(lim, 1e-3)

    plt.figure(figsize=(7, 6))
    plt.scatter(x_true, y_pred, s=6, alpha=0.5, edgecolors="none")
    plt.plot([-lim, lim], [-lim, lim], "k-", linewidth=1.5)
    plt.xlim([-lim, lim]); plt.ylim([-lim, lim])
    plt.xlabel("DFT Force (eV/Å)")
    plt.ylabel("NN Force (eV/Å)")
    plt.title(f"{title}\nMAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"[saved] {out_png}  |  MAE={mae:.6f} RMSE={rmse:.6f} R²={r2:.6f}")


def build_model(device):
    """Make sure these hyperparams match your training run."""
    model = E3NNForceModel(
        num_atom_types=120,
        cutoff=4.0,                 # ↓ neighborhood size → big speedup; still OK for smoke tests
        num_rbf=16,                 # ↓ basis size
        max_l=1,                    # only scalars + vectors; drops costly higher-order tensors
        irreps_hidden="8x0e + 4x1o",# narrow hidden width
        num_layers=2                # minimal depth that still learns something
    ).to(device)
    return model


def load_weights(model, ckpt_path, device="cpu", strict=True):
    """Robust checkpoint loader that handles full checkpoints and raw state_dicts."""
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Case A: full training checkpoint with metadata
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    # Case B: raw state_dict saved directly
    elif isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
        state = checkpoint
    else:
        raise RuntimeError(f"Unrecognized checkpoint format. Top-level keys: {list(checkpoint.keys())[:10]}")

    # Strip 'module.' if needed (DDP/DataParallel)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    # Load
    load_result = model.load_state_dict(state, strict=strict)

    # If non-strict, print diagnostics
    if not strict and isinstance(load_result, tuple):
        missing, unexpected = load_result
        print(f"[load_weights] missing={missing}\n[load_weights] unexpected={unexpected}")

    return checkpoint  # caller can use epoch/val_loss if present


def evaluate_all(model_path, index_csv, root_dir, out_dir,
                 aggregate_png, dpi=200, make_per_sample=False, per_sample_prefix="forces_parity_",
                 strict_load=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    dataset = LazyGraphDataset(index_csv=index_csv, root_dir=root_dir)
    n = len(dataset)
    print(f"Found {n} test samples in {index_csv}")

    # Model + weights
    model = build_model(device)
    print(f"Loading weights from {model_path} …")
    _ckpt = load_weights(model, model_path, device=device, strict=strict_load)
    model.eval()

    # Aggregate buffers
    all_true = []
    all_pred = []

    # Read index once (may or may not have a 'tag' column)
    df_idx = pd.read_csv(index_csv)
    tags = df_idx["tag"].tolist() if "tag" in df_idx.columns else [None] * n

    os.makedirs(out_dir, exist_ok=True)

    for i in range(n):
        data = dataset[i].to(device)
        # We need grads for force = -∇_pos E
        data.pos.requires_grad_(True)

        # Forward + forces
        pred_E = model(data)                      # scalar or [batch], your model should return energy
        pred_F = compute_forces(pred_E, data.pos) # (N_atoms, 3)

        # Move to CPU numpy
        pred_F_np = pred_F.detach().cpu().numpy()
        true_F_np = data.y_forces.cpu().numpy()

        # Append aggregate
        all_pred.append(pred_F_np.reshape(-1))
        all_true.append(true_F_np.reshape(-1))

        # Optional per-sample parity plot
        if make_per_sample:
            tag = tags[i] if i < len(tags) else None
            suffix = f"sample{i}"
            if tag is not None and isinstance(tag, str):
                safe_tag = "".join(c if c.isalnum() or c in "-_." else "_" for c in tag)
                suffix = f"{suffix}_{safe_tag}"
            out_png = os.path.join(out_dir, f"{per_sample_prefix}{suffix}.png")
            forces_parity_plot(
                x_true=true_F_np.reshape(-1),
                y_pred=pred_F_np.reshape(-1),
                title=f"Force Parity — sample {i}" + (f" | Tag: {tag}" if tag else ""),
                out_png=out_png,
                dpi=dpi,
                show=False,
            )

        # Free graph asap
        del pred_E, pred_F
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if (i + 1) % 20 == 0 or i == n - 1:
            print(f"Processed {i+1}/{n}")

    # Make aggregate plot
    x_all = np.concatenate(all_true, axis=0)
    y_all = np.concatenate(all_pred, axis=0)
    agg_png = os.path.join(out_dir, aggregate_png)
    forces_parity_plot(
        x_true=x_all,
        y_pred=y_all,
        title="Force Parity — ALL test samples",
        out_png=agg_png,
        dpi=dpi,
        show=False,
    )


if __name__ == "__main__":
    evaluate_all(
        model_path        = CONFIG["model_path"],
        index_csv         = CONFIG["index_csv"],
        root_dir          = CONFIG["root_dir"],
        out_dir           = CONFIG["out_dir"],
        aggregate_png     = CONFIG["aggregate_png"],
        dpi               = CONFIG["dpi"],
        make_per_sample   = CONFIG["make_per_sample"],
        per_sample_prefix = CONFIG["per_sample_prefix"],
        strict_load       = CONFIG["strict_load"],
    )
