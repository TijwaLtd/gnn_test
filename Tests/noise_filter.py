import os
import sys
import glob
import random
import numpy as np
import pandas as pd
import torch

# ----------------- USER CONFIG -----------------
# Add one or more roots to sweep. Use raw strings for Windows.
ROOT_DIRS = [
    r"D:\Sara\All_DFT_Data",
]

GRAPH_FILE_NAME = "graphs.pt"  # the file to look for in each leaf folder
SEEDS = [0, 42, 123, 999, 2025]  # "different random generators"
CHECKPOINT = r"C:\Users\labadmin\Documents\GNN\GNN-ML-Model\500_epochs\checkpoints\e3nn_epoch_0500.pth"  # <<< set to your trained weights
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_CSV = "noise_scores.csv"
OUT_FILTERED = "filtered_structure_ids.txt"
FORCE_HEAD_PRESENT = True  # set False if your checkpoint/model has no force head
STRICT_LOAD = False  # True if checkpoint must match exactly

# NEW: Position jitter settings
POSITION_JITTER_STD = 1e-3  # Å-scale jitter for sensitivity testing

# ----------------- REPRO/SEEDING -----------------
def set_seed(seed: int):
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

# ----------------- MODEL / IMPORTS -----------------
# Your model should live here. Adjust the class name/ctor if needed.
from e3nn_gnn_model import E3NNForceModel  # <<< change if your class name differs

def build_model():
    # If your model takes hyperparameters, set them here so it matches the checkpoint.
    model = E3NNForceModel()
    return model

def load_checkpoint(model, ckpt_path, strict=STRICT_LOAD):
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=strict)
    return model

def setup_mc_dropout(model):
    """
    Enable Monte Carlo dropout: keep dropout active but BatchNorm in eval mode
    """
    model.train()  # Enable dropout for MC inference
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            m.eval()  # Keep BN in eval so stats don't drift
    return model

# ----------------- DATA DISCOVERY/LOADING -----------------
def find_graph_files(roots, file_name):
    files = []
    for root in roots:
        # ** recursively ** find all graphs.pt under this root
        pattern = os.path.join(root, "**", file_name)
        matches = glob.glob(pattern, recursive=True)
        files.extend(matches)
    # stable order for reproducibility
    files = sorted(set(os.path.normpath(f) for f in files))
    return files

def load_as_pyg_list(path):
    """
    Load a graphs.pt that may store:
    - a single PyG Data object
    - a list/tuple of Data
    - a dict with key 'data' or 'list'
    Returns a list of Data objects.
    """
    obj = torch.load(path, map_location="cpu")
    
    try:
        from torch_geometric.data import Data
    except Exception:
        Data = None

    # Single Data
    if Data is not None and isinstance(obj, Data):
        return [obj]
    
    # List/tuple of Data
    if isinstance(obj, (list, tuple)):
        return list(obj)
    
    # Dict
    if isinstance(obj, dict):
        if "data" in obj:
            x = obj["data"]
            if Data is not None and isinstance(x, Data):
                return [x]
            if isinstance(x, (list, tuple)):
                return list(x)
        if "list" in obj and isinstance(obj["list"], (list, tuple)):
            return list(obj["list"])
    
    # Fallback: try to interpret as one structure
    return [obj]

def structure_id_from_path(root_dirs, file_path):
    """
    Produce a stable ID like '120\\B101C19_1\\S_128317' (root-relative folder).
    """
    file_path = os.path.normpath(file_path)
    # strip the filename
    folder = os.path.dirname(file_path)
    
    # choose the shortest root-relative path (in case multiple roots overlap)
    candidates = []
    for rd in root_dirs:
        rd = os.path.normpath(rd)
        if folder.lower().startswith(rd.lower()):
            rel = os.path.relpath(folder, rd)
            candidates.append(os.path.join(os.path.basename(rd), rel))
    
    if candidates:
        return candidates[0]
    
    # else just return the last two levels
    parts = folder.split(os.sep)
    return os.path.join(*parts[-3:]) if len(parts) >= 3 else folder

def apply_position_jitter(data, jitter_std=POSITION_JITTER_STD):
    """
    Apply small random perturbations to atomic positions for sensitivity testing
    """
    if hasattr(data, "pos") and data.pos is not None:
        data.pos = data.pos + jitter_std * torch.randn_like(data.pos)
    return data

# ----------------- FORWARD (single structure) -----------------
def forward_once(model, data, device, compute_forces=True):
    """
    Returns (energy_scalar, forces_tensor_or_None) for ONE structure.
    If the model doesn't output forces, we compute them via autograd: F = -∂E/∂pos
    """
    # Apply position jitter for sensitivity testing
    data = apply_position_jitter(data)
    
    # move to device
    data = data.to(device)

    # Try the simple "model(data)" API first
    try:
        out = model(data)
    except TypeError:
        # Fall back to explicit signature if needed
        x = getattr(data, "x", None)
        edge_index = getattr(data, "edge_index", None)
        edge_attr = getattr(data, "edge_attr", None)
        pos = getattr(data, "pos", None)
        batch = getattr(data, "batch", None)
        out = model(x, edge_index, edge_attr, pos, batch)

    # Normalize energy
    if isinstance(out, dict):
        energy = out.get("energy", None)
        forces = out.get("forces", None)  # some models return forces directly
    else:
        energy, forces = out, None

    if energy is None:
        return None, None

    energy = torch.atleast_1d(energy).mean()

    # If we want forces and the model didn't return them, compute via autograd
    if compute_forces and (forces is None):
        # enable grads on positions ONLY
        # NOTE: do NOT wrap this in torch.no_grad() / inference_mode()
        pos = data.pos
        need_restore = pos.requires_grad is False
        if need_restore:
            pos = pos.detach().clone().requires_grad_(True)
        
        # update the data object view so the model uses grad-enabled pos
        data.pos = pos
        
        # re-run forward with grad-enabled pos
        try:
            out2 = model(data)
        except TypeError:
            x = getattr(data, "x", None)
            edge_index = getattr(data, "edge_index", None)
            edge_attr = getattr(data, "edge_attr", None)
            batch = getattr(data, "batch", None)
            out2 = model(x, edge_index, edge_attr, pos, batch)

        energy = out2["energy"] if isinstance(out2, dict) else out2
        energy = torch.atleast_1d(energy).mean()

        forces = -torch.autograd.grad(
            outputs=energy,
            inputs=data.pos,
            grad_outputs=None,
            retain_graph=False,
            create_graph=False,
            allow_unused=False
        )[0]  # shape: (N_atoms, 3)

        # detach to keep memory clean
        forces = forces.detach()
        
        # (optional) restore original pos without grads
        if need_restore:
            data.pos = data.pos.detach()

    return energy.detach(), forces

# ----------------- MAIN: SEED SWEEP -----------------
def main():
    graph_files = find_graph_files(ROOT_DIRS, GRAPH_FILE_NAME)
    if not graph_files:
        print("[error] No graphs.pt files found. Check ROOT_DIRS and GRAPH_FILE_NAME.")
        sys.exit(1)
    
    print(f"[info] Found {len(graph_files)} graph files (e.g., first: {graph_files[0]})")

    energy_store = {}    # sid -> [seed runs...]
    force_norm_store = {}  # sid -> [seed runs...]

    for seed in SEEDS:
        print(f"\n[seed] {seed}")
        set_seed(seed)
        
        model = build_model().to(DEVICE)
        model = load_checkpoint(model, CHECKPOINT).to(DEVICE)
        
        # Enable Monte Carlo dropout instead of model.eval()
        model = setup_mc_dropout(model)

        for fp in graph_files:
            sid = structure_id_from_path(ROOT_DIRS, fp)
            data_list = load_as_pyg_list(fp)

            # If a file packs multiple graphs (e.g., periodic images), average their predictions
            e_vals = []
            f_norm_vals = []

            for data in data_list:
                energy, forces = forward_once(model, data, DEVICE, compute_forces=True)
                
                if energy is not None:
                    e_vals.append(float(energy.detach().cpu().item()))
                
                if forces is not None:
                    f = forces.cpu().numpy()
                    per_atom_l2 = np.linalg.norm(f, axis=-1).mean()
                    f_norm_vals.append(float(per_atom_l2))

            # Aggregate for this file (if multiple sub-graphs existed)
            if e_vals:
                energy_store.setdefault(sid, []).append(float(np.mean(e_vals)))
            if f_norm_vals:
                force_norm_store.setdefault(sid, []).append(float(np.mean(f_norm_vals)))

    # ---------- aggregate across seeds ----------
    rows = []
    all_sids = sorted(set(list(energy_store.keys()) + list(force_norm_store.keys())))
    
    for sid in all_sids:
        e = energy_store.get(sid, [])
        f = force_norm_store.get(sid, [])
        
        e_mean = np.mean(e) if e else np.nan
        e_std = np.std(e) if e else np.nan
        f_mean = np.mean(f) if f else np.nan
        f_std = np.std(f) if f else np.nan
        
        rows.append(dict(
            structure_id=sid,
            energy_mean=e_mean,
            energy_std=e_std,
            force_mean=f_mean,
            force_std=f_std
        ))

    df = pd.DataFrame(rows)
    
    # Combined noise score: conservative max of available stds
    if FORCE_HEAD_PRESENT and df["force_std"].notna().any():
        df["noise_score"] = df[["energy_std", "force_std"]].fillna(0.0).max(axis=1)
    else:
        df["noise_score"] = df["energy_std"].fillna(0.0)

    df.sort_values("noise_score", ascending=False, inplace=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"[done] wrote {OUT_CSV} with {len(df)} rows")

    # IMPROVED IQR cutoff for outliers
    q1, q3 = df["noise_score"].quantile([0.25, 0.75])
    iqr = q3 - q1
    
    if not np.isfinite(iqr) or iqr <= 0:
        cutoff = float(df["noise_score"].quantile(0.95))
        print(f"[info] IQR failed, using 95th percentile: {cutoff:.6f}")
    else:
        cutoff = q3 + 1.5 * iqr
        print(f"[info] cutoff (IQR rule): {cutoff:.6f}")

    noisy = df[df["noise_score"] > cutoff]["structure_id"].tolist()
    
    with open(OUT_FILTERED, "w") as f:
        for sid in noisy:
            f.write(f"{sid}\n")
    
    print(f"[done] flagged {len(noisy)} noisy structures → {OUT_FILTERED}")

if __name__ == "__main__":
    main()
