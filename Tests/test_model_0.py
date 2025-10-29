"""
plot_forces_parity_all.py (epsilon helper)
------------------------------------------
Evaluates a trained E3NN force model over ALL rows in index_test.csv and:
  1) Prints per-sample and global statistics of predicted force magnitudes.
  2) Dumps all predicted forces (components and norms) to CSVs.
  3) (keeps) aggregate parity plot over all samples.

Use the global percentile table to pick a good epsilon for the EQ penalty.
"""

import os, sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# USER CONFIG – EDIT HERE
# ---------------------------------------------------------------------
CONFIG = dict(
    model_path        = r"GNN-ML-Model/models/100_epoch_model.pth",
    index_csv         = r"C:/Users/labadmin/Documents/GNN/GNN-ML-Model/index_test.csv",
    root_dir          = r"D:/Sara/All_DFT_Data/Test_Data",
    out_dir           = "parity_plots",          # folder to write outputs
    aggregate_png     = "forces_parity_ALL.png",
    dpi               = 300,
    make_per_sample   = False,                   # per-sample parity plots
    per_sample_prefix = "forces_parity_",
    strict_load       = True,

    # NEW: file names for dumps
    dump_components_csv = "pred_forces_components.csv",
    dump_norms_csv      = "pred_force_norms.csv",
)
# ---------------------------------------------------------------------

# If this file lives inside /Code/ML/, make parent visible in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from e3nn_gnn_model import E3NNForceModel, compute_forces
from lazy_graph_dataset import LazyGraphDataset


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def forces_parity_plot(x_true: np.ndarray, y_pred: np.ndarray,
                       title: str,
                       out_png: str,
                       dpi: int = 200,
                       show: bool = False):
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
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
        state = checkpoint
    else:
        raise RuntimeError(f"Unrecognized checkpoint format. Top-level keys: {list(checkpoint.keys())[:10]}")

    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    load_result = model.load_state_dict(state, strict=strict)
    if not strict and isinstance(load_result, tuple):
        missing, unexpected = load_result
        print(f"[load_weights] missing={missing}\n[load_weights] unexpected={unexpected}")
    return checkpoint


def evaluate_all(model_path, index_csv, root_dir, out_dir,
                 aggregate_png, dpi=200, make_per_sample=False, per_sample_prefix="forces_parity_",
                 strict_load=True,
                 dump_components_csv="pred_forces_components.csv",
                 dump_norms_csv="pred_force_norms.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    dataset = LazyGraphDataset(index_csv=index_csv, root_dir=root_dir)
    n = len(dataset)
    print(f"Found {n} test samples in {index_csv}")

    # Model + weights
    model = build_model(device)
    print(f"Loading weights from {model_path} …")
    _ = load_weights(model, model_path, device=device, strict=strict_load)
    model.eval()

    # For parity plot
    all_true = []
    all_pred = []

    # For dumps
    rows_components = []  # one row per atom-component
    rows_norms      = []  # one row per atom

    # Optional tags
    df_idx = pd.read_csv(index_csv)
    tags = df_idx["tag"].tolist() if "tag" in df_idx.columns else [None] * n

    os.makedirs(out_dir, exist_ok=True)

    for i in range(n):
        data = dataset[i].to(device)
        data.pos.requires_grad_(True)

        # Forward + forces
        pred_E = model(data)                      # energy
        pred_F = compute_forces(pred_E, data.pos) # (N_atoms, 3)

        # To numpy (CPU)
        pred_F_np = pred_F.detach().cpu().numpy()
        true_F_np = data.y_forces.cpu().numpy()

        # Aggregate for parity
        all_pred.append(pred_F_np.reshape(-1))
        all_true.append(true_F_np.reshape(-1))

        # ---- Collect rows for CSV dumps ----
        tag = tags[i] if i < len(tags) else None
        n_atoms = pred_F_np.shape[0]
        # components table
        for a in range(n_atoms):
            rows_components.append({
                "sample_idx": i,
                "tag": tag,
                "atom_idx": a,
                "Fx_pred": pred_F_np[a, 0],
                "Fy_pred": pred_F_np[a, 1],
                "Fz_pred": pred_F_np[a, 2],
                # include DFT for reference if wanted
                "Fx_dft": true_F_np[a, 0],
                "Fy_dft": true_F_np[a, 1],
                "Fz_dft": true_F_np[a, 2],
            })
        # norms table (vector magnitude)
        norms_pred = np.linalg.norm(pred_F_np, axis=1)
        norms_dft  = np.linalg.norm(true_F_np, axis=1)
        for a in range(n_atoms):
            rows_norms.append({
                "sample_idx": i,
                "tag": tag,
                "atom_idx": a,
                "Fnorm_pred": norms_pred[a],
                "Fnorm_dft": norms_dft[a],
            })

        # ---- Per-sample printout (concise stats) ----
        p = np.percentile(norms_pred, [0, 50, 90, 95, 99, 99.5, 99.9, 100])
        print(f"[sample {i:03d}] atoms={n_atoms}  "
              f"min={p[0]:.6f}  median={p[1]:.6f}  P95={p[3]:.6f}  P99={p[4]:.6f}  "
              f"P99.9={p[6]:.6f}  max={p[7]:.6f}  (eV/Å)")

        # Optional per-sample parity plot
        if make_per_sample:
            safe_tag = None
            if tag is not None and isinstance(tag, str):
                safe_tag = "".join(c if c.isalnum() or c in "-_." else "_" for c in tag)
            suffix = f"sample{i}" if safe_tag is None else f"sample{i}_{safe_tag}"
            out_png = os.path.join(out_dir, f"{per_sample_prefix}{suffix}.png")
            forces_parity_plot(
                x_true=true_F_np.reshape(-1),
                y_pred=pred_F_np.reshape(-1),
                title=f"Force Parity — sample {i}" + (f" | Tag: {tag}" if tag else ""),
                out_png=out_png,
                dpi=dpi,
                show=False,
            )

        # cleanup
        del pred_E, pred_F
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if (i + 1) % 20 == 0 or i == n - 1:
            print(f"Processed {i+1}/{n}")

    # ---------- Save CSV dumps ----------
    components_df = pd.DataFrame(rows_components)
    norms_df      = pd.DataFrame(rows_norms)

    comp_path = os.path.join(out_dir, CONFIG["dump_components_csv"])
    norms_path = os.path.join(out_dir, CONFIG["dump_norms_csv"])
    components_df.to_csv(comp_path, index=False)
    norms_df.to_csv(norms_path, index=False)
    print(f"[saved] {comp_path}  ({len(components_df)} rows)")
    print(f"[saved] {norms_path}   ({len(norms_df)} rows)")

    # ---------- Global percentiles (useful for epsilon) ----------
    x_all = np.concatenate(all_true, axis=0)
    y_all = np.concatenate(all_pred, axis=0)
    # predicted vector norms across ALL atoms of ALL samples
    all_pred_vectors = np.vstack([y.reshape(-1, 3) for y in [y_all.reshape(-1, 3)]])[0]
    all_pred_norms = np.linalg.norm(all_pred_vectors, axis=1)
    pct = [50, 75, 90, 95, 97.5, 99, 99.5, 99.9, 100]
    vals = np.percentile(all_pred_norms, pct)
    print("\n=== GLOBAL percentiles of |F_pred| (eV/Å) across ALL EQ atoms ===")
    for p, v in zip(pct, vals):
        print(f"P{p:<5}: {v:.6f}")
    print("Suggested ε candidates: near P95–P99 (start with P95).")

    # ---------- Aggregate parity plot ----------
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
        dump_components_csv = CONFIG["dump_components_csv"],
        dump_norms_csv      = CONFIG["dump_norms_csv"],
    )
