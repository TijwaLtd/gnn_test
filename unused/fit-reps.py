import os
import torch
import numpy as np
from torch_geometric.data import DataLoader

ROOT = "D:\Sara\All_DFT_Data"
all_graphs = []
for dp, _, files in os.walk(ROOT):
    if "graphs.pt" in files:
        full = os.path.join(dp, "graphs.pt")
        lst = torch.load(full, map_location="cpu")
        all_graphs.extend(lst if isinstance(lst, list) else [lst])

loader = DataLoader(all_graphs, batch_size=1, shuffle=False)
E_raw, nB_list, nC_list = [], [], []
for data in loader:
    E_raw.append(float(data.raw_energy))
    z = data.z
    nB_list.append(int((z==5).sum()))
    nC_list.append(int((z==6).sum()))

# Build design matrix with an intercept column of ones
A = np.vstack([nB_list, nC_list, np.ones_like(nB_list)]).T  # shape (N,3)
b = np.array(E_raw)

# Solve for [E_B, E_C, C0]
(E_ref_B, E_ref_C, C0), *_ = np.linalg.lstsq(A, b, rcond=None)

print("Fitted references:")
print(f"  E_B = {E_ref_B:.6f} eV")
print(f"  E_C = {E_ref_C:.6f} eV")
print(f"  E_0_per_atom  = {C0:.6f} eV")
