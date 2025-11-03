# test_residual_distribution.py

import os
import torch
import numpy as np
import pytest
from torch_geometric.data import DataLoader

# ── Paste your new fitted values here ────────────────────────────────────────
E_ref_B         = -6.387798   # from fit_refs.py
E_ref_C         = -10.022939  # from fit_refs.py
E0_per_atom     =  0.097160   # from fit_refs.py

@pytest.fixture(scope="module")
def all_graphs():
    ROOT = "DFT_data"
    graphs = []
    for dp, _, files in os.walk(ROOT):
        if "graphs.pt" in files:
            lst = torch.load(os.path.join(dp, "graphs.pt"), map_location="cpu")
            graphs.extend(lst if isinstance(lst, list) else [lst])
    assert graphs, f"No graphs found under {ROOT}"
    return graphs

def test_residuals_stats(all_graphs):
    loader = DataLoader(all_graphs, batch_size=1, shuffle=False)
    residuals = []
    for data in loader:
        E_raw     = float(data.raw_energy)
        z         = data.z
        nB        = int((z == 5).sum())
        nC        = int((z == 6).sum())
        n_atoms   = data.n_atoms.item()

        # new baseline: E_B*nB + E_C*nC + E0_per_atom * n_atoms
        E_baseline = nB*E_ref_B + nC*E_ref_C + n_atoms*E0_per_atom
        y_res = (E_raw - E_baseline) / n_atoms
        residuals.append(y_res)

    res = np.array(residuals)
    mean, std = res.mean(), res.std()
    print(f"\nResiduals per atom: mean={mean:.4f} eV, std={std:.4f} eV")

    assert abs(mean) < 1e-3, f"Mean residual {mean:.4f} too far from zero"
    assert std < 0.5,       f"Std dev {std:.4f} unexpectedly large"
