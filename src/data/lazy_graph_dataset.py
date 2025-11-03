import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Data
from functools import lru_cache

class LazyGraphDataset(Dataset):
    def __init__(self, index_csv, root_dir=".", transform=None, limit_samples=None):
        """
        index_csv: CSV with at least 'graph_path'; optional 'root_dir', 'is_eq',
                   'n_B','n_C','composition','tag', 'n_atoms', etc.
        root_dir:  Fallback root (used if a row lacks 'root_dir')
        transform: Optional callable(Data) -> Data
        limit_samples: Optional int, if provided, limits the dataset to this many samples.
        """
        self.df = pd.read_csv(index_csv)

        # Normalize expected columns
        if "is_eq" not in self.df.columns:
            self.df["is_eq"] = 0
        # make sure graph_path is string
        self.df["graph_path"] = self.df["graph_path"].astype(str)

        if limit_samples is not None and limit_samples < len(self.df):
            df_eq = self.df[self.df['is_eq'] == 1]
            df_neq = self.df[self.df['is_eq'] == 0]

            num_eq = len(df_eq)
            num_neq = len(df_neq)
            total_original = num_eq + num_neq

            if total_original == 0: # Handle empty dataframe case
                self.df = pd.DataFrame()
            else:
                # Calculate proportional limits
                limit_eq = int(limit_samples * (num_eq / total_original))
                limit_neq = limit_samples - limit_eq

                # Ensure at least one of each if possible and limit_samples allows
                if limit_eq == 0 and num_eq > 0 and limit_samples > 0:
                    limit_eq = 1
                    limit_neq = max(0, limit_samples - 1)
                if limit_neq == 0 and num_neq > 0 and limit_samples > 0:
                    limit_neq = 1
                    limit_eq = max(0, limit_samples - 1)
                
                # Sample from each group
                sampled_eq = df_eq.sample(n=min(limit_eq, num_eq), random_state=42)
                sampled_neq = df_neq.sample(n=min(limit_neq, num_neq), random_state=42)

                self.df = pd.concat([sampled_eq, sampled_neq]).sample(frac=1, random_state=42).reset_index(drop=True)
                print(f"Dataset limited to {len(self.df)} samples (EQ: {len(sampled_eq)}, NEQ: {len(sampled_neq)})")

        self.default_root = os.path.abspath(root_dir)
        self.transform = transform

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
