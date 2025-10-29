import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Data
from functools import lru_cache

class LazyGraphDataset(Dataset):
    def __init__(self, index_csv, root_dir=".", transform=None):
        """
        index_csv: CSV with at least 'graph_path'; optional 'root_dir', 'is_eq',
                   'n_B','n_C','composition','tag', 'n_atoms', etc.
        root_dir:  Fallback root (used if a row lacks 'root_dir')
        transform: Optional callable(Data) -> Data
        """
        self.df = pd.read_csv(index_csv)
        self.default_root = os.path.abspath(root_dir)
        self.transform = transform

        # Normalize expected columns
        if "is_eq" not in self.df.columns:
            self.df["is_eq"] = 0
        # make sure graph_path is string
        self.df["graph_path"] = self.df["graph_path"].astype(str)

    def __len__(self):
        return len(self.df)

    @lru_cache(maxsize=2048)
    def _load_graph(self, full_path_norm: str):
        """Load a graph given a *normalized absolute* path (cache key-safe)."""
        graph_obj = torch.load(full_path_norm, map_location="cpu", weights_only=False)
        return graph_obj[0] if isinstance(graph_obj, list) else graph_obj

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Pick per-row root_dir if present; else fall back
        row_root = row["root_dir"] if "root_dir" in row and isinstance(row["root_dir"], str) else self.default_root
        full_path = os.path.join(row_root, row["graph_path"])
        # Normalize to a canonical absolute path for caching
        full_path = os.path.abspath(full_path).replace("\\", "/")

        # Load graph
        data: Data = self._load_graph(full_path)

        # ---- attach metadata (guard missing cols) ----
        # equilibrium flag for training mask (bool[1] so PyG stacks to [num_graphs])
        # robust: handles ints or strings like "0"/"1"
        val = row.get("is_eq", 0)              # can be 0/1 or "0"/"1"
        try:
            is_eq = bool(int(val))
        except Exception:
            is_eq = False
        data.is_eq = torch.tensor([is_eq], dtype=torch.bool)

        # optional numeric/meta fields
        for col in ("n_B", "n_C", "n_atoms"):
            if col in row:
                try:
                    setattr(data, col, int(row[col]))
                except Exception:
                    setattr(data, col, row[col])

        for col in ("composition", "tag", "size_bin"):
            if col in row:
                setattr(data, col, row[col])

        # Keep a reference to source path if helpful for debugging
        data.graph_path = row["graph_path"]
        data.root_dir = row_root

        # ---- optional user transform ----
        if self.transform:
            data = self.transform(data)

        return data
