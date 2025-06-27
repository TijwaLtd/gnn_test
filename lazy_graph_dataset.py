import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch.serialization import add_safe_globals
from functools import lru_cache

# Allow safe deserialization of PyG Data objects
add_safe_globals([Data])

class LazyGraphDataset(Dataset):
    def __init__(self, index_csv, root_dir, transform=None):
        """
        Parameters:
        - index_csv: Path to the index.csv file
        - root_dir: Root path containing all the subfolders with graphs.pt
        - transform: Optional transform to apply to each Data object
        """
        self.df = pd.read_csv(index_csv)
        self.root_dir = os.path.abspath(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    @lru_cache(maxsize=1024)
    def _load_graph(self, rel_path):
        full_path = os.path.join(self.root_dir, rel_path)
        graph_list = torch.load(full_path, map_location="cpu", weights_only=False)

        # Support for either single graph or list of graphs
        return graph_list[0] if isinstance(graph_list, list) else graph_list

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row['graph_path']
        graph = self._load_graph(rel_path)

        # Optional: attach metadata if useful
        graph.n_B = row['n_B']
        graph.n_C = row['n_C']
        graph.composition = row['composition']
        graph.tag = row['tag']

        if self.transform:
            graph = self.transform(graph)

        return graph
