from torch_geometric.loader import DataLoader
from lazy_graph_dataset import LazyGraphDataset

dataset = LazyGraphDataset(
    index_csv="/Users/muhammadzainasad/Documents/Documents - Muhammad’s MacBook Air/Research Internship/Code/DFT_data/DFT_data/index.csv",
    root_dir="/Users/muhammadzainasad/Documents/Documents - Muhammad’s MacBook Air/Research Internship/Code/DFT_data/DFT_data"
)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

for batch in loader:
    print(batch)
    break