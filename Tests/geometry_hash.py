import os, sys, hashlib
import numpy as np
import pandas as pd
import torch

# ---------- CONFIG ----------
INDEX_CSV = r"C:\Users\labadmin\Documents\GNN\GNN-ML-Model\index.csv"   # your combined CSV
ROOT_DIR  = r"D:\Sara\All_DFT_Data"                           # same root used by LazyGraphDataset
CUTOFF    = 6.4     # Å: ignore pair distances > cutoff (keeps signature compact)
TOL       = 1e-3    # Å: rounding tolerance for the signature
OUT_ALL   = "geom_hashes_all.csv"       # per-row hashes
OUT_FAM   = "geom_families_gt1.csv"     # families with count>1
# ----------------------------

# Make parent visible if this script sits in /Code/ML/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lazy_graph_dataset import LazyGraphDataset   # your loader

def _cartesian_positions(data):
    """Return Nx3 cartesian positions in Å. Adjust if you store fractional coords."""
    if hasattr(data, "pos"):
        return data.pos.detach().cpu().numpy()
    raise AttributeError("Data object has no .pos field; adapt this function to your data format.")

def geom_hash_from_graph(data, cutoff=CUTOFF, tol=TOL):
    """Rotation/translation/permutation-robust signature from pairwise distances."""
    pos = _cartesian_positions(data)  # (N,3)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    tri = d[np.triu_indices(len(pos), k=1)]
    tri = tri[(tri > 1e-9) & (tri <= cutoff)]
    sig = np.round(tri / tol).astype(np.int32)
    sig.sort()
    return hashlib.sha1(sig.tobytes()).hexdigest()

def main():
    df = pd.read_csv(INDEX_CSV).reset_index(drop=True)
    # Your screenshot shows column 'graph_path'; keep handy for reporting
    path_col = "graph_path" if "graph_path" in df.columns else None
    tag_col  = "tag" if "tag" in df.columns else None

    ds = LazyGraphDataset(index_csv=INDEX_CSV, root_dir=ROOT_DIR)

    rows = []
    for i in range(len(ds)):
        h = geom_hash_from_graph(ds[i], cutoff=CUTOFF, tol=TOL)
        rows.append({
            "row_idx": i,
            "geom_hash": h,
            "graph_path": (df.loc[i, path_col] if path_col else None),
            "tag": (df.loc[i, tag_col] if tag_col else None),
        })
        if (i+1) % 50 == 0 or i == len(ds)-1:
            print(f"[hash] {i+1}/{len(ds)}")

    hashes = pd.DataFrame(rows)
    hashes.to_csv(OUT_ALL, index=False)
    print(f"[saved] {OUT_ALL} ({len(hashes)} rows)")

    fam = hashes.groupby("geom_hash").agg(
        count=("row_idx", "count"),
        example_idx=("row_idx", "first"),
        example_path=("graph_path", "first"),
        example_tag=("tag", "first"),
    ).reset_index()

    dup_fam = fam[fam["count"] > 1].sort_values("count", ascending=False)
    dup_fam.to_csv(OUT_FAM, index=False)
    print(f"[saved] {OUT_FAM} ({len(dup_fam)} families with count>1)")

    # Console summary
    total = len(hashes)
    unique = fam.shape[0]
    dups = len(dup_fam)
    print("\n=== Geometry-family summary ===")
    print(f"Total rows           : {total}")
    print(f"Unique geom families : {unique}")
    print(f"Families with >1 rows: {dups}")
    if dups:
        print("\nTop overlapping families:")
        print(dup_fam.head(10).to_string(index=False))
        print("\n[Action] To avoid leakage in a split, keep each geom_hash entirely in TRAIN or TEST (grouped split).")
    else:
        print("[OK] No repeated geometry families detected; random split won’t leak by geometry.")

if __name__ == "__main__":
    main()
