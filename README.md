Project Overview

This repository implements a pipeline to process VASP DFT outputs, build
an indexed graph dataset, and train an equivariant GNN (E₃NN) to predict
energies and forces.

Repository Structure

bulk_export_all.py \# Traverses directories, parses OUTCAR and POSCAR
files DFT_processor_2\_Zain.py \# Custom parser/extractor of energies,
forces, and atomic positions gaussian_rbf.py \# Radial basis (Gaussian)
expansion for interatomic distances generate_index.py \# Walks DFT_data
folder, loads graphs.pt, computes B/C counts, writes index.csv
lazy_graph_dataset.py \# PyG Dataset: lazy-loads graph objects per entry
in index.csv e3nn_gnn_model.py \# Defines E3NN equivariant GNN for
energy+force prediction train.py \# Training & evaluation script using
DataLoader + loss/optimizer README.md \# This document LICENSE \#
Project license DFT_data/ \# Folder containing generated graphs and
index.csv \<composition\>/\<structure\>/graphs.pt index.csv \# Generated
by \`generate_index.py\`

## 1\. DFT Processing & Graph Construction

1\. bulk_export_all.py: Recursively finds all calculation folders under
DFT_data/, reads VASP OUTCAR & CONTCAR (or POSCAR) to extract:  - Total
energy  - Per-atom forces  - Atomic numbers & positions  - Cell lattice
vectors It then serializes each structure into a PyG graph object
(graphs.pt) saving node features (z, pos), edge connectivity, and target
tensors (y_energy, y_forces).

2\. DFT_processor_2\_Zain.py: Custom logic for parsing VASP file
formats, handling multiple convergence steps, and cleaning coordinate
units.

3\. gaussian_rbf.py: Implements RBFExpansion:  - Generates num_rbf
Gaussian kernels centered in \[0, cutoff\].  - Applies smooth cosine
cutoff.  - Returns edge-level RBF features of shape \[num_edges,
num_rbf\].

## 2\. Dataset Indexing & Lazy Loading

1\. generate_index.py:  - Walks through DFT_data/ directory.  - Loads
each graphs.pt with PyTorch (including custom PyG safe globals).  -
Counts number of B and C atoms (via data.z) to form composition tags
like B47C13.  - Writes index.csv with columns:  - graph_path (relative
to DFT_data/)  - n_B, n_C, composition, tag

2\. lazy_graph_dataset.py:  - Reads index.csv.  - Implements a
torch.utils.data.Dataset that on \_\_getitem\_\_:  - Loads the
corresponding graphs.pt lazily.  - Attaches additional fields (n_B, n_C,
composition, tag).  - Allows efficient batching via PyG DataLoader
without preloading all graphs into memory.

## 3\. Equivariant GNN Model (E₃NN)

1\. e3nn_gnn_model.py:  - Imports from e3nn:  - o3.Irreps for
irreducible representation specifications.  -
FullyConnectedTensorProduct, Linear for equivariant layers.  - Gate,
NormActivation for gated non-linearities.  - Builds an E3NNForceModel
class:  - Edge features: RBF + Spherical Harmonics (via separate
RBFExpansion & SphericalHarmonics).  - Message passing: Equivariant
tensor products over edges.  - Gated activations keeping equivariance.
 - Output heads:
    - Scalar energy prediction per node, summed to
graph-level energy.
     - Force prediction via gradient F = -∇ₓE.
     - HENCE THE MODEL PREDICTS ENERGY AND THEN COMPUTES F AS THE NEGATIVE GRADIENT OF ENERGY
## In short:

- First predict E.

- Then compute F = –∇E using the predicted energy and the input positions.

- Finally compare those forces to your original DFT forces to train/validate the model.

2\. Key design choices:  - irreps_hidden controls numbers of
scalar/vector/tensor channels (e.g. 32x0e+16x1o+8x2e).  - max_l sets max
spherical harmonic degree.  - num_layers layers of equivariant graph
convolutions.

## 4\. Training & Evaluation

1\. train.py:  - Loads LazyGraphDataset and wraps in DataLoader.  -
Instantiates E3NNForceModel, moves to cuda if available.  - Defines
combined loss: MSE(E) + λ·MSE(F) with force gradient computation via
torch.autograd.grad.  - Optimizer: AdamW + ReduceLROnPlateau, optional
early stopping.

2\. Workflow:  - Split dataset (e.g. 80/20 train/val).  - Loop over
epochs:  - train_one_epoch(...)  - evaluate_model(...)  - Save best
checkpoint (best_e3nn_model.pth).


