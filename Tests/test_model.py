"""
test_single_file.py
-------------------
Run a trained E3NN force model on ONE structure and print an energy/force
comparison.

Edit the CONFIG block below to point to your files.
"""

import torch
import numpy as np
import os, sys
import pandas as pd

# ---------------------------------------------------------------------
# USER CONFIG – EDIT HERE
# ---------------------------------------------------------------------
CONFIG = dict(  
    model_path = r"C:\Users\labadmin\Documents\GNN\GNN-ML-Model\models\100_epoch_model.pth",  # .pth checkpoint
    index_csv  = "C:/Users/labadmin/Documents/GNN/GNN-ML-Model/index_test.csv",               # CSV with relative paths
    root_dir   = "D:/Sara/All_DFT_Data/Test_Data",                         # folder containing samples
    sample_idx = 100-2,                                  # 1704 onwards
)   
# ---------------------------------------------------------------------

# If this file lives inside /Code/ML/, make parent visible in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from e3nn_gnn_model import E3NNForceModel, compute_forces
from lazy_graph_dataset import LazyGraphDataset


def test_single_file(model_path, index_csv, root_dir, sample_idx):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load dataset & grab the chosen sample
    print(f"Loading dataset from {index_csv} …")
    dataset = LazyGraphDataset(index_csv=index_csv, root_dir=root_dir)
    if not 0 <= sample_idx < len(dataset):
        raise IndexError(
            f"sample_idx {sample_idx} out of range 0-{len(dataset)-1}"
        )

    data = dataset[sample_idx].to(device)
    print(f"Testing sample {sample_idx}: {getattr(data, 'composition', 'N/A')}")

    # 2. Build model (make sure these hyper-params match training!)
    model = E3NNForceModel(
        num_atom_types=120,        # FIXME: set to your real # of species
        cutoff=6.3,
        num_rbf=64,
        max_l=3,
        irreps_hidden="32x0e + 16x1o + 8x2e + 4x3o",
        num_layers=3,
    ).to(device)

    # 3. Load checkpoint weights
    print(f"Loading weights from {model_path} …")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 4. Forward pass + forces
    data.pos.requires_grad_(True)
    pred_energy = model(data)                   # shape: scalar tensor
    pred_forces = compute_forces(pred_energy, data.pos)

    # 5. Detach & convert to NumPy for reporting
    pred_energy = pred_energy.item()
    true_energy = data.y_energy.item()
    pred_forces = pred_forces.detach().cpu().numpy()
    true_forces = data.y_forces.cpu().numpy()
    
    tag = pd.read_csv(index_csv).loc[sample_idx, "tag"]   # column name is 'tag'
    print(f"Testing sample {sample_idx}: {getattr(data, 'composition', 'N/A')}")
    print(f"Tag: {tag}")

    print("Energy")
    energy_rmse = np.sqrt((pred_energy - true_energy) ** 2)
    print(f"RMSE: {energy_rmse:.6f}")
    print("Forces")
    mae = np.mean(np.abs(pred_forces - true_forces))
    print(f"MAE = {mae:.6f} eV/Å")

    rmse = np.sqrt(np.mean((pred_forces - true_forces) ** 2))
    print(f"RMSE = {rmse:.6f} eV/Å\n") 

    print("\n--- Energy ---")
    print(f"Predicted: {pred_energy:.6f}  |  True: {true_energy:.6f}")
    print(f"Abs error: {abs(pred_energy - true_energy):.6f}")
    energy_rmse = np.sqrt((pred_energy - true_energy) ** 2)
    print(f"RMSE     : {energy_rmse:.6f}")


    print("\n--- Forces (MAE over all atoms/components) ---")
    mae = np.mean(np.abs(pred_forces - true_forces))
    print(f"MAE = {mae:.6f} eV/Å\n")

    rmse = np.sqrt(np.mean((pred_forces - true_forces) ** 2))
    print(f"RMSE = {rmse:.6f} eV/Å\n")


    header = "{:<5} {:>11} {:>11} {:>11} | {:>11} {:>11} {:>11}"
    print(header.format("Atom", "Pred Fx", "Pred Fy", "Pred Fz",
                        "True Fx", "True Fy", "True Fz"))
    print("-"*80)
    for i, (pf, tf) in enumerate(zip(pred_forces, true_forces)):
        print(f"{i:<5} {pf[0]:11.6f} {pf[1]:11.6f} {pf[2]:11.6f} | "
              f"{tf[0]:11.6f} {tf[1]:11.6f} {tf[2]:11.6f}")


if __name__ == "__main__":
    # Run with CONFIG values
    test_single_file(
        model_path = CONFIG["model_path"],
        index_csv  = CONFIG["index_csv"],
        root_dir   = CONFIG["root_dir"],
        sample_idx = CONFIG["sample_idx"],
    )

    # -----------------------------------------------------------------
    # Optional: keep CLI overrides (uncomment if you still want them)
    # -----------------------------------------------------------------
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model",  default=CONFIG["model_path"])
    # parser.add_argument("--index",  default=CONFIG["index_csv"])
    # parser.add_argument("--root",   default=CONFIG["root_dir"])
    # parser.add_argument("--sample", default=CONFIG["sample_idx"], type=int)
    # args = parser.parse_args()
    # test_single_file(args.model, args.index, args.root, args.sample)
