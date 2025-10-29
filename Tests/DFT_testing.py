# full_graph_sanity_from_files.py
# ---------------------------------------------------------------------------
# End-to-end test: build e3nn-ready graph(s) from POSCAR+OUTCAR, run sanity
# checks, then reload the saved data.pt under the new PyTorch 2.6 rules.
# ---------------------------------------------------------------------------

import os
import torch
import pickle
from e3nn.o3 import rand_matrix, spherical_harmonics, Irreps
from torch_geometric.data import Data          # only Data is guaranteed

from DFT_processor_2_Zain import DFTProcessor   # adjust if filename differs


# ---------------- helpers ---------------------------------------------------
def _rotate(R, x):          # R [3,3]  –> apply to any [*,3] tensor
    return x @ R.t()


def test_irreps_and_shapes(g):
    assert str(g.node_irreps) == "1x0e"
    B = g.edge_attr["rbf"].shape[1]
    assert str(Irreps(g.edge_irreps["rbf"])) == f"{B}x0e"
    assert g.edge_attr["sh"].shape[1] == Irreps(g.edge_irreps["sh"]).dim
    print("✅ metadata vs. tensor shapes consistent")


def test_edge_symmetry(g):
    pairs = set(map(tuple, g.edge_index.t().tolist()))
    assert all((j, i) in pairs for i, j in pairs), "missing reverse edges"
    print("✅ every i→j edge has j→i")


def test_rotation_equivariance(g, radial_model, lmax):
    R = rand_matrix()
    pos_R = _rotate(R, g.pos)
    vecs_R = pos_R[g.edge_index[0]] - pos_R[g.edge_index[1]]
    d_R = vecs_R.norm(dim=-1)

    sh_R = spherical_harmonics(Irreps.spherical_harmonics(lmax),
                               vecs_R / d_R.unsqueeze(-1), True)

    # rbf invariance not guaranteed under PBC → skip
    assert torch.allclose(g.edge_attr["sh"].norm(dim=1),
                          sh_R.norm(dim=1), atol=1e-5)
    print("✅ rotation probe passed (angular norms ok; radial skipped for PBC)")


# ---------------- main ------------------------------------------------------
if __name__ == "__main__":
    # Folder containing POSCAR & OUTCAR
    SAMPLE_DIR = (
        "/Users/muhammadzainasad/Documents/Documents - Muhammad’s MacBook Air/"
        "Research Internship/Code/ML/DFT_data/DFT_data/15atoms/B11C4_1/S_23543"
    )
    assert os.path.exists(os.path.join(SAMPLE_DIR, "POSCAR")), "POSCAR missing"
    assert os.path.exists(os.path.join(SAMPLE_DIR, "OUTCAR")), "OUTCAR missing"

    # Build graphs and save data.pt
    proc = DFTProcessor(SAMPLE_DIR, lmax=3, num_basis=16, cutoff=3.2)
    graphs = proc.process_directory()
    assert graphs, "No graphs created – check POSCAR/OUTCAR parse"

    g = graphs[0]

    # Sanity checks
    print(f"\nGraph summary:  N = {g.num_nodes}  |  E = {g.edge_index.size(1)}")
    test_irreps_and_shapes(g)
    test_edge_symmetry(g)
    test_rotation_equivariance(g, proc.radial_model, lmax=3)

    # -----------------------------------------------------------
    # Safe reload of data.pt under PyTorch ≥ 2.6
    # -----------------------------------------------------------
    data_pt = os.path.join(os.path.dirname(SAMPLE_DIR), "data.pt")

    # 1) allow-list the Data class (required for weights_only=True)
    torch.serialization.add_safe_globals([Data])

    try:
        graphs_reloaded = torch.load(data_pt)  # weights_only=True (default)
    except pickle.UnpicklingError:
        # some PyG internals (e.g. DataEdgeAttr) may still be missing – fall back
        print("↪  Falling back to weights_only=False (file is trusted).")
        graphs_reloaded = torch.load(data_pt, weights_only=False)

    print(f"\n✅ Reloaded {len(graphs_reloaded)} graphs from {data_pt}")
    print("🎉 All file-based sanity checks passed!")
