# Traverses graph.pt and writes a row per file from index.csv
# exports graph_path, n_B, n_C, composition, tag

import os
import torch
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data
from torch.serialization import add_safe_globals

torch.serialization.add_safe_globals([Data])

def count_elements(z_list):
    n_B = (z_list == 5).sum().item()
    n_C = (z_list == 6).sum().item()
    return n_B, n_C

def build_index(root_dir, output_csv="index.csv"):
    rows = []

    for dirpath, _, filenames in os.walk(root_dir):
        if "graphs.pt" in filenames:
            full_path = os.path.join(dirpath, "graphs.pt")
            try:
                data_list = torch.load(full_path, map_location="cpu", weights_only=False)

                # Handle case: single graph or list of graphs
                if isinstance(data_list, list):
                    data = data_list[0]  # take the first one
                else:
                    data = data_list     # already a single graph

                n_B, n_C = count_elements(data.z)
                composition = f"B{n_B}C{n_C}"
                relative_path = os.path.relpath(full_path, root_dir)
                tag = os.path.basename(os.path.dirname(full_path))

                rows.append({
                    "graph_path": relative_path,
                    "n_B": n_B,
                    "n_C": n_C,
                    "composition": composition,
                    "tag": tag,
                })

            except Exception as e:
                print(f"❌ Failed to read {full_path}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(root_dir, output_csv), index=False)
    print(f"✅ Wrote index with {len(df)} entries to {output_csv}")


if __name__ == "__main__":
    root_dataset = "/Users/muhammadzainasad/Documents/Documents - Muhammad’s MacBook Air/Research Internship/Code/ML/DFT_data/DFT_data"
    build_index(root_dataset)
