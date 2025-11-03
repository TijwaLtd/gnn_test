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
import argparse

# Check for a command-line flag to set the memory allocator config
if '--use-expandable-segments' in sys.argv:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    print("Using expandable segments for CUDA memory allocation.")
    # Remove the flag from sys.argv so it doesn't interfere with other argument parsing
    sys.argv.remove('--use-expandable-segments')

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

# ---------------------------------------------------------------------
# USER CONFIG
# ---------------------------------------------------------------------
CONFIG = dict(
    model_path        = r"models/best_model.pth", # <-- IMPORTANT: UPDATE THIS PATH
    index_csv         = r"data/index_val.csv",    # <-- Or your test set
    root_dir          = r"data/DFT_DATA",
    out_dir           = "parity_plots",
    aggregate_png     = "forces_parity_ALL.png",
    dpi               = 300,
    make_per_sample   = True, # Set to True to get EQ/NEQ plots
    per_sample_prefix = "forces_parity_",
    strict_load       = True,
)
# ---------------------------------------------------------------------

# If this file lives inside /Code/ML/, make parent visible in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models.e3nn_gnn_model import E3NNForceModel, compute_forces
from src.data.lazy_graph_dataset import LazyGraphDataset


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
        cutoff=6.3,
        num_rbf=64,
        max_l=3,
        irreps_hidden="32x0e + 16x1o + 8x2e + 4x3o",
        num_layers=3,
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
                 strict_load=True, gen_all_plots=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    dataset = LazyGraphDataset(index_csv=index_csv, root_dir=root_dir)
    n = len(dataset)
    print(f"Found {n} test samples in {index_csv}")

    # Create DataLoader
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Model + weights
    model = build_model(device)
    print(f"Loading weights from {model_path} …")
    _ckpt = load_weights(model, model_path, device=device, strict=strict_load)
    model.eval()

    # Aggregate buffers
    all_true, all_pred = [], []
    eq_true, eq_pred = [], []
    neq_true, neq_pred = [], []

    os.makedirs(out_dir, exist_ok=True)

    processed_samples_count = 0
    for batch_data in data_loader: # 'batch_data' is a Batch object
        batch_data = batch_data.to(device)
        batch_data.pos.requires_grad_(True)

        # Forward + forces
        pred_E = model(batch_data)
        pred_F = compute_forces(pred_E, batch_data.pos)

        # Move to CPU numpy
        pred_F_np = pred_F.detach().cpu().numpy()
        true_F_np = batch_data.y_forces.cpu().numpy()

        # Iterate through individual graphs within the batch
        num_graphs_in_batch = batch_data.num_graphs
        for graph_in_batch_idx in range(num_graphs_in_batch):
            start_node_idx = batch_data.ptr[graph_in_batch_idx]
            end_node_idx = batch_data.ptr[graph_in_batch_idx + 1]

            # Forces for the current graph
            graph_pred_F_np = pred_F_np[start_node_idx:end_node_idx].flatten()
            graph_true_F_np = true_F_np[start_node_idx:end_node_idx].flatten()

            # Append to aggregate lists
            all_pred.append(graph_pred_F_np)
            all_true.append(graph_true_F_np)

            # Get tag and is_eq_flag for the current graph
            is_eq_for_graph = batch_data.is_eq[graph_in_batch_idx].item()
            tag_for_graph = batch_data.tag[graph_in_batch_idx] if hasattr(batch_data, 'tag') else None

            if is_eq_for_graph:
                eq_pred.append(graph_pred_F_np)
                eq_true.append(graph_true_F_np)
            else:
                neq_pred.append(graph_pred_F_np)
                neq_true.append(graph_true_F_np)

            # Optional per-sample parity plot
            if make_per_sample and gen_all_plots:
                suffix = f"sample{processed_samples_count}"
                if tag_for_graph is not None and isinstance(tag_for_graph, str):
                    safe_tag = "".join(c if c.isalnum() or c in "-_." else "_" for c in tag_for_graph)
                    suffix = f"{suffix}_{safe_tag}"
                out_png = os.path.join(out_dir, f"{per_sample_prefix}{suffix}.png")
                forces_parity_plot(
                    x_true=graph_true_F_np,
                    y_pred=graph_pred_F_np,
                    title=f"Force Parity — sample {processed_samples_count}" + (f" | Tag: {tag_for_graph}" if tag_for_graph else ""),
                    out_png=out_png,
                    dpi=dpi,
                    show=False,
                )

            processed_samples_count += 1

        del pred_E, pred_F
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if (processed_samples_count % 20 == 0 or processed_samples_count == n):
            print(f"Processed {processed_samples_count}/{n}")

    # --- Generate Aggregate Plots ---

    # 1) All samples
    x_all = np.concatenate(all_true)
    y_all = np.concatenate(all_pred)
    agg_png = os.path.join(out_dir, aggregate_png)
    forces_parity_plot(
        x_true=x_all, y_pred=y_all,
        title="Force Parity — ALL test samples",
        out_png=agg_png, dpi=dpi
    )

    if gen_all_plots:
        # 2) EQ samples only
        if eq_true:
            x_eq = np.concatenate(eq_true)
            y_eq = np.concatenate(eq_pred)
            eq_png = os.path.join(out_dir, "forces_parity_EQ.png")
            forces_parity_plot(
                x_true=x_eq, y_pred=y_eq,
                title="Force Parity — EQUILIBRIUM samples",
                out_png=eq_png, dpi=dpi
            )

        # 3) NEQ samples only
        if neq_true:
            x_neq = np.concatenate(neq_true)
            y_neq = np.concatenate(neq_pred)
            neq_png = os.path.join(out_dir, "forces_parity_NEQ.png")
            forces_parity_plot(
                x_true=x_neq, y_pred=y_neq,
                title="Force Parity — NON-EQUILIBRIUM samples",
                out_png=neq_png, dpi=dpi
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained E3NN force model.")
    parser.add_argument('--gen-all-plots', action='store_true', help='Generate all parity plots (per-sample, EQ, NEQ).')
    args = parser.parse_args()

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
        gen_all_plots     = args.gen_all_plots
    )