import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import DataLoader

# Read index_clean.csv
df = pd.read_csv("index_clean.csv")
ROOT = "D:/Sara/All_DFT_Data"

all_graphs = []
for _, row in df.iterrows():
    graph_path = row['graph_path']
    full = os.path.join(ROOT, graph_path)
    lst = torch.load(full, map_location="cpu")
    all_graphs.extend(lst if isinstance(lst, list) else [lst])

loader = DataLoader(all_graphs, batch_size=1, shuffle=False)
E_raw, nB_list, nC_list, n_atoms_list = [], [], [], []
for data in loader:
    E_raw.append(float(data.raw_energy))
    z = data.z
    nB_list.append(int((z == 5).sum()))
    nC_list.append(int((z == 6).sum()))
    n_atoms_list.append(int(data.n_atoms.item()))

# Build design matrix with columns [nB, nC, n_atoms]
A = np.vstack([nB_list, nC_list, n_atoms_list]).T   # shape (N,3)
b = np.array(E_raw)

# Solve for [E_B, E_C, E0_per_atom]
(E_ref_B, E_ref_C, E0_per_atom), *_ = np.linalg.lstsq(A, b, rcond=None)

print("Fitted references:")
print(f"  E_B             = {E_ref_B:.6f} eV")
print(f"  E_C             = {E_ref_C:.6f} eV")
print(f"  E0_per_atom     = {E0_per_atom:.6f} eV/atom")
