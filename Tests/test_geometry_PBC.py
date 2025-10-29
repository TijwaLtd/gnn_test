# test_model_geometry.py

import os
import torch
import pytest
from torch_geometric.loader import DataLoader

from e3nn_gnn_model import E3NNForceModel, compute_forces

# ──────────────────────────────────────────────────────────────────────────────
# Point this at your real graphs.pt
DATA_DIR  = os.path.dirname(__file__)
GRAPHS_PT = os.path.join(
    DATA_DIR,
    "DFT_data",
    "15atoms",
    "B11C4_1",
    "S_23543",
    "graphs.pt"
)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def graphs():
    assert os.path.exists(GRAPHS_PT), f"Cannot find graphs.pt at {GRAPHS_PT}"
    return torch.load(GRAPHS_PT)

def get_model():
    return E3NNForceModel(
        num_atom_types=10,
        cutoff=6.3,  
        num_rbf=16,  
        max_l=3,     
        irreps_hidden="32x0e + 16x1o + 8x2e + 4x3o",
        num_layers=3
    ).eval()

def test_model_forward_and_forces(graphs):
    model = get_model()
    loader = DataLoader(graphs, batch_size=1, shuffle=False)
    batch = next(iter(loader))
    batch.pos.requires_grad_(True)

    energy = model(batch)
    assert energy.shape == (1,), f"Expected energy.shape==(1,), got {energy.shape}"
    assert torch.isfinite(energy).all()

    forces = compute_forces(energy, batch.pos)
    assert forces.shape == batch.pos.shape, \
        f"Expected forces.shape=={batch.pos.shape}, got {forces.shape}"
    assert torch.isfinite(forces).all()

def test_edge_shift_values(graphs):
    data = graphs[0]
    assert hasattr(data, "edge_shift"), "Data object missing edge_shift"
    # at least one edge has a nonzero shift
    assert (data.edge_shift.abs().sum(dim=1) > 0).any(), "No periodic edges found"

def test_pbc_invariance(graphs):
    model = get_model()
    data0 = graphs[0].clone()
    data0.pos.requires_grad_(True)

    energy0 = model(data0)
    forces0 = compute_forces(energy0, data0.pos)

    # translate by one lattice vector (along the first basis)
    shift_vec = data0.lattice[0]  # [3]
    data1 = data0.clone()
    data1.pos = data1.pos + shift_vec
    data1.pos.requires_grad_(True)

    energy1 = model(data1)
    forces1 = compute_forces(energy1, data1.pos)

    # energies should be identical under a full-cell translation
    assert torch.allclose(energy1, energy0, atol=1e-5), \
        f"Energy changed under PBC shift: {energy1.item()} vs {energy0.item()}"

    # forces should also match
    assert torch.allclose(forces1, forces0, atol=1e-5), "Forces changed under PBC shift"
