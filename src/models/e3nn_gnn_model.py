
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import radius_graph
from torch_scatter import scatter
import numpy as np
import torch.optim as Optimizer

import os,sys
import csv

import e3nn
from e3nn import o3
from e3nn.o3 import Irreps, FullyConnectedTensorProduct, Linear
from e3nn.nn import NormActivation
import torch.nn.functional as F

# Step 1 : TANHU cutoff 
# Re-run 
# ----------- DONE ----------------
# Step 2: Kalman Filter
# Step 3: Optuna

#IMPORTANT , enbeddings meaning for weights , why are there 64 what each number means




# Constants
E_ref_B     = -0.809536
E_ref_C     = -4.607478
E0_per_atom = -5.417014

os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)


class Tee:
    def __init__(self, logfile_path):
        self.terminal = sys.stdout
        self.log = open(logfile_path, 'a', encoding='utf-8', errors='replace')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Tee('logs/terminal.txt')
sys.stderr = sys.stdout


# -----------------------------------------------------------------------------
# I/O UTILITIES
# -----------------------------------------------------------------------------

def _ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d)

config = {
    # total number of epochs you’ll run
    "num_epochs":     500,

    # how often (in epochs) to do the heavy energy/force dumps
    "dump_interval":   50,
}

def save_embeddings_cumulative(model, filename, epoch):
    """
    Append Boron & Carbon embedding vectors for this epoch into one file.
    Format:
      Epoch | Element | w0 | w1 | ... | w{D-1}
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    W = model.atom_embedding.weight.detach().cpu().numpy()  # [num_types, D]
    boron   = W[5].tolist()
    carbon  = W[6].tolist()
    dim     = W.shape[1]
    exists  = os.path.exists(filename)

    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        if not exists:
            header = ['Epoch','Element'] + [f'w{i}' for i in range(dim)]
            writer.writerow(header)
        writer.writerow([epoch, 'Boron',  *boron])
        writer.writerow([epoch, 'Carbon', *carbon])

def save_energy_data(model, loader, filename, device):
    """
    StructureID | NumAtoms | DFT_Energy | NN_Energy
    """
    _ensure_dir(filename)
    model.eval()
    with open(filename, 'w', newline='') as f, torch.no_grad():
        writer = csv.writer(f, delimiter='|')
        writer.writerow(['StructureID','NumAtoms','DFT_Energy','NN_Energy'])

        for batch in loader:
            batch = batch.to(device)
            batch.pos.requires_grad_(True)

            # 1) predict NN total energies
            pred_E = model(batch).view(-1)        # [batch_size]

            # 2) per‐graph sizes
            atom2graph = batch.batch              # [total_atoms]
            n_graphs   = pred_E.size(0)

            # 3) count atoms per graph
            ones       = torch.ones_like(atom2graph, dtype=torch.float)
            N          = scatter(ones, atom2graph, dim=0, dim_size=n_graphs)

            # 4) count B and C atoms per graph
            isB        = (batch.z == 5).float()
            isC        = (batch.z == 6).float()
            nB         = scatter(isB, atom2graph, dim=0, dim_size=n_graphs)
            nC         = scatter(isC, atom2graph, dim=0, dim_size=n_graphs)

            # 5) reconstruct absolute DFT energy
            E_base     = nB * E_ref_B + nC * E_ref_C + N * E0_per_atom
            y_res      = batch.y_energy_res         # per‐structure residual
            dft_E      = (y_res * N + E_base).cpu().numpy()

            # 6) collect NN predictions
            nn_E       = pred_E.cpu().numpy()
            N_np       = N.cpu().numpy().astype(int)

            # 7) structure IDs
            sids = getattr(batch, 'structure_id', list(range(n_graphs)))

            for sid, nat, de, ne in zip(sids, N_np, dft_E, nn_E):
                writer.writerow([sid, nat, de, ne])

def save_force_data(model, loader, filename, device):
    """
    StructureID | AtomID | DFT_Fx| DFT_Fy| DFT_Fz | NN_Fx| NN_Fy| NN_Fz
    """
    _ensure_dir(filename)
    model.eval()
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerow(['StructureID','AtomID',
                         'DFT_Fx','DFT_Fy','DFT_Fz',
                         'NN_Fx','NN_Fy','NN_Fz'])

        for batch in loader:
            batch = batch.to(device)
            # turn on grad for positions
            batch.pos.requires_grad_(True)

            # 1) Forward pass (no torch.no_grad)
            E_pred = model(batch).view(-1)  # [batch_size]

            # 2) Backward for forces
            F_pred = -torch.autograd.grad(
                E_pred.sum(),
                batch.pos,
                retain_graph=False,
                create_graph=False
            )[0].cpu().numpy()              # [total_atoms,3]

            # 3) Ground-truth forces
            F_dft = batch.y_forces.cpu().numpy()  # [total_atoms,3]

            # 4) Map atoms → graph index
            atom2graph = batch.batch.cpu().numpy()

            for atom_idx, (g, fd, fn) in enumerate(zip(atom2graph, F_dft, F_pred)):
                writer.writerow([g, atom_idx, *fd, *fn])

def save_embeddings(model, filename):
    """
    Extracts learned embedding for atomic numbers 5 (B) and 6 (C)
    """
    _ensure_dir(filename)
    W = model.atom_embedding.weight.detach().cpu().numpy()  # [num_types,embed_dim]
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerow(['Element','EmbedVector'])
        writer.writerow(['Boron', *W[5].tolist()])
        writer.writerow(['Carbon', *W[6].tolist()])

def save_epoch_metrics(epoch, te, tf, ve, vf, filename):
    """
    Appends a line: Epoch | Train_E_RMSE | Train_F_RMSE | Val_E_RMSE | Val_F_RMSE
    """
    _ensure_dir(filename)
    header = ['Epoch','Train_E_RMSE','Train_F_RMSE','Val_E_RMSE','Val_F_RMSE']
    exists = os.path.exists(filename)
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        if not exists:
            writer.writerow(header)
        writer.writerow([epoch, te, tf, ve, vf])

class RBFExpansion(nn.Module):
    """Radial Basis Function expansion for edge distances"""
    def __init__(self, num_rbf=20, cutoff=5.0):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        # Centers for RBF functions
        # self.centers = nn.Parameter(torch.linspace(0, cutoff, num_rbf), requires_grad=False)
        # self.widths = nn.Parameter(torch.full((num_rbf,), cutoff / num_rbf), requires_grad=False)
        self.register_buffer("centers", torch.linspace(0, cutoff, num_rbf))
        self.register_buffer("widths",  torch.full((num_rbf,), cutoff/num_rbf))


    
    def forward(self, distances):
        """
        Args:
            distances: [num_edges] edge distances
        Returns:
            rbf_features: [num_edges, num_rbf] RBF expansion
        """
        # Gaussian RBF
        rbf = torch.exp(-((distances.unsqueeze(1) - self.centers) / self.widths)**2)
        # Apply cutoff function (cosine cutoff)
        cutoff_mask = (distances < self.cutoff).float()
        cutoff_smooth = 0.5 * (torch.cos(torch.pi * distances / self.cutoff) + 1) * cutoff_mask
        return rbf * cutoff_smooth.unsqueeze(1)
    
class SphericalHarmonics(nn.Module):
    """Compute spherical harmonics for edge directions"""
    def __init__(self, max_l=3):
        super().__init__()
        self.max_l = max_l
        self.irreps_sh = o3.Irreps.spherical_harmonics(max_l)
    
    def forward(self, edge_vec):
        """
        Args:
            edge_vec: [num_edges, 3] normalized edge vectors
        Returns:
            sh_features: [num_edges, num_sh] spherical harmonics
        """
        return o3.spherical_harmonics(self.irreps_sh, edge_vec, normalize=True)

class EquivariantGraphLayer(nn.Module):
    """Single equivariant message passing layer with proper tensor products"""
    def __init__(self, irreps_node_input, irreps_node_hidden, irreps_edge_attr, num_rbf):
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_hidden = o3.Irreps(irreps_node_hidden)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        
        # Tensor product for message construction
        self.tp = FullyConnectedTensorProduct(
            self.irreps_node_input,
            self.irreps_edge_attr,
            self.irreps_node_hidden,
            shared_weights=False,
            internal_weights=False
        )

        # MLP for processing RBF features to generate TP weights
        self.rbf_mlp = nn.Sequential(
            nn.Linear(num_rbf, num_rbf),
            nn.SiLU(),
            nn.Linear(num_rbf, self.tp.weight_numel)
        )
        
        # Self-connection for residual
        if self.irreps_node_input == self.irreps_node_hidden:
            self.self_connection = nn.Identity()
        else:
            self.self_connection = Linear(self.irreps_node_input, self.irreps_node_hidden)
        
    def forward(self, node_features, edge_index, edge_attr, rbf_features):
        """
        Args:
            node_features: [num_nodes, irreps_node_input] node features
            edge_index: [2, num_edges] edge connectivity
            edge_attr: [num_edges, irreps_edge_attr] edge attributes (SH)
            rbf_features: [num_edges, num_rbf] RBF features
        Returns:
            updated_node_features: [num_nodes, irreps_node_hidden]
        """
        source, target = edge_index
        
        # Construct messages via tensor product
        rbf_weights = self.rbf_mlp(rbf_features)
        messages = self.tp(node_features[source], edge_attr, rbf_weights)  # [num_edges, tp_irreps]
        
        # Aggregate messages at target nodes
        num_nodes = node_features.size(0)
        aggregated_messages = scatter(messages, target, dim=0, dim_size=num_nodes, reduce='sum')
        
        # Add self-connection (residual)
        self_contribution = self.self_connection(node_features)
        
        return aggregated_messages + self_contribution
    
class E3NNForceModel(nn.Module):
    """Complete E3NN equivariant model for energy and force prediction"""
    def __init__(
        self,
        num_atom_types=100,
        cutoff=5.0,
        num_rbf=20,
        max_l=3,
        irreps_hidden="32x0e + 16x1o + 8x2e",
        num_layers=3,
        num_heads=4
    ):
        super().__init__()
        self.cutoff = cutoff
        self.num_atom_types = num_atom_types
        
        # Edge feature computation
        self.rbf_expansion = RBFExpansion(num_rbf=num_rbf, cutoff=cutoff)
        self.spherical_harmonics = SphericalHarmonics(max_l=max_l)
        
        # Node embeddings - start with larger embedding
        self.atom_embedding = nn.Embedding(num_atom_types, 64)
        
        # Initial node features (scalars only)
        irreps_node_input = "64x0e"  # 64 scalar features
        irreps_edge_attr = self.spherical_harmonics.irreps_sh
        
        # Convert to proper irreps format
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        
        # Input projection
        self.input_projection = Linear(irreps_node_input, self.irreps_hidden)
        
        # Message passing layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                layer_input = self.irreps_hidden
            else:
                layer_input = self.irreps_hidden
                
            layer = EquivariantGraphLayer(
                irreps_node_input=layer_input,
                irreps_node_hidden=self.irreps_hidden,
                irreps_edge_attr=irreps_edge_attr,
                num_rbf=num_rbf
            )
            self.layers.append(layer)
        
        # Gated nonlinearities between layers
        self.gates = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.gates.append(NormActivation(self.irreps_hidden, scalar_nonlinearity=torch.nn.functional.silu))
        
        # Output head for energy (scalar prediction only)
        self.energy_head = nn.Sequential(
            Linear(self.irreps_hidden, self.irreps_hidden),
            NormActivation(self.irreps_hidden, scalar_nonlinearity=torch.nn.functional.silu),
            Linear(self.irreps_hidden, o3.Irreps("1x0e"))
        )
        
    def forward(self, data):
        """
        Args:
            data: PyG data object with pos, z, batch, edge_index (optional)
        Returns:
            energy: [batch_size] predicted energies per graph
        """
        pos = data.pos  # [num_nodes, 3]
        atomic_numbers = data.z  # [num_nodes]
        
        # Assume data is always a Batch object
        batch = data.batch  # [num_nodes]
        num_graphs = data.num_graphs
        
        # — use precomputed PBC edges and shifts from the Data object —
        edge_index, edge_shift = data.edge_index, data.edge_shift      # [2,E], [E,3]
        source, target = edge_index
        lattice = data.lattice                                          # [num_graphs, 3, 3]

        # If edge_batch is not directly available, derive it from node batch
        if not hasattr(data, 'edge_batch'):
            data.edge_batch = data.batch[edge_index[0]]
        
        # edge_batch maps each edge to its graph index in the batch
        lattice_for_edges = lattice[data.edge_batch] # [E_total, 3, 3]

        # Now, perform batch matrix multiplication
        shifted_pos = edge_shift * lattice_for_edges # [E_total, 3]

        edge_vec = (pos[target] + shifted_pos) - pos[source] # [E_total,3]
        edge_length = edge_vec.norm(dim=1)                              # [E]

        
        # Compute edge features
        rbf_features = self.rbf_expansion(edge_length)  
        # SH module normalizes internally, so pass the raw vectors
        sh_features  = self.spherical_harmonics(edge_vec)

        
        # Initial node features from atomic numbers
        node_features = self.atom_embedding(atomic_numbers).float()  # [num_nodes, 64]
        
        # Project to hidden irreps
        node_features = self.input_projection(node_features)  # [num_nodes, irreps_hidden]
        
        # Message passing layers with gated nonlinearities
        for i, layer in enumerate(self.layers):
            node_features = layer(node_features, edge_index, sh_features, rbf_features)
            
            # Apply gated nonlinearity (except for last layer)
            if i < len(self.layers) - 1:
                node_features = self.gates[i](node_features)
        
        # Predict energy per node using only scalar features
        node_energies = self.energy_head(node_features).squeeze(-1)  # [num_nodes]
        
        # Sum over nodes in each graph to get total energy
        if batch is not None:
            graph_energies = scatter(node_energies, batch, dim=0, reduce='sum')  # [batch_size]
        else:
            # Single graph case
            graph_energies = node_energies.sum(dim=0, keepdim=True)
        
        return graph_energies
    
def compute_forces(energy, pos, allow_unused=True):
    """
    Compute forces as negative gradient of energy with respect to positions
    Args:
        energy: [batch_size] predicted energies
        pos: [num_nodes, 3] atomic positions (with requires_grad=True)
    Returns:
        forces: [num_nodes, 3] predicted forces
    """
    # Handle both single graphs and batches
    if energy.dim() == 0:
        energy_sum = energy
    else:
        energy_sum = energy.sum()
        
    forces = -torch.autograd.grad(
        outputs=energy_sum,
        inputs=pos,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=allow_unused
    )[0]
    return forces

def train_one_epoch(model, loader, optimizer, device, lambda_forces=1.0):
    """Training loop with hybrid force loss (MSE for NEQ, Huber for EQ)."""
    model.train()
    total_loss = 0.0
    energy_loss_total = 0.0
    force_loss_total = 0.0
    
    # Define separate loss functions
    mse_loss_fn = nn.MSELoss()
    huber_loss_fn = nn.HuberLoss(delta=0.1) # delta is a tunable hyperparameter

    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        batch.pos.requires_grad_(True)
        
        try:
            # Predict total energy and forces
            pred_energy = model(batch)
            pred_forces = compute_forces(pred_energy, batch.pos)

            # --- Residual-based energy loss (MSE) ---
            z = batch.z
            nB = (z == 5).sum().float()
            nC = (z == 6).sum().float()
            n_atoms = batch.n_atoms.float()
            E_base = nB * E_ref_B + nC * E_ref_C + n_atoms * E0_per_atom
            y_res = batch.y_energy_res
            pred_res = (pred_energy - E_base) / n_atoms
            energy_loss = mse_loss_fn(pred_res, y_res)
            
            # --- Hybrid Force Loss ---
            force_loss = torch.tensor(0.0, device=device)
            if hasattr(batch, 'y_forces'):
                true_forces = batch.y_forces
                
                # Create masks for equilibrium and non-equilibrium atoms
                if hasattr(batch, "batch"):
                    eq_node_mask = batch.is_eq[batch.batch]
                else: # single-graph case
                    eq_node_mask = batch.is_eq.new_full((batch.pos.size(0),), bool(batch.is_eq.item()))
                
                neq_node_mask = ~eq_node_mask

                # Calculate loss for non-equilibrium atoms (MSE)
                if neq_node_mask.any():
                    force_loss_neq = mse_loss_fn(pred_forces[neq_node_mask], true_forces[neq_node_mask])
                    force_loss += force_loss_neq

                # Calculate loss for equilibrium atoms (Huber)
                if eq_node_mask.any():
                    force_loss_eq = huber_loss_fn(pred_forces[eq_node_mask], true_forces[eq_node_mask])
                    force_loss += force_loss_eq
            
            # Total batch loss
            total_batch_loss = energy_loss + lambda_forces * force_loss

            # Backprop and optimization
            optimizer.zero_grad()
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate losses for reporting
            batch_size = getattr(batch, "num_graphs", 1)
            total_loss += total_batch_loss.item() * batch_size
            energy_loss_total += energy_loss.item() * batch_size
            force_loss_total += force_loss.item() * batch_size

        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    # Compute averages
    num_samples = len(loader.dataset)
    avg_total_loss = total_loss / num_samples
    avg_energy_loss = energy_loss_total / num_samples
    avg_force_loss = force_loss_total / num_samples
    
    # Return losses (without the old eq_penalty)
    return avg_total_loss, avg_energy_loss, avg_force_loss, 0.0 # Return 0 for old penalty

def evaluate_model(model, loader, device):
    """
    Evaluation loop computing RMSE and MAE for energy and forces.
    Now uses a hybrid loss logic for evaluation metrics.
    """
    model.eval()
    
    mse_loss_fn = nn.MSELoss(reduction='sum')
    huber_loss_fn = nn.HuberLoss(delta=0.1, reduction='sum')

    total_energy_loss = 0.0
    total_force_loss = 0.0
    num_graphs = 0
    num_atoms_neq = 0
    num_atoms_eq = 0

    for batch in loader:
        batch = batch.to(device)
        batch.pos.requires_grad_(True)

        # 1) Forward pass
        pred_energy = model(batch).view(-1)

        # 2) Energy loss
        with torch.no_grad():
            z = batch.z
            nB = (z == 5).sum().float()
            nC = (z == 6).sum().float()
            N = batch.n_atoms.float()
            E_base = nB * E_ref_B + nC * E_ref_C + N * E0_per_atom
            y_res = batch.y_energy_res
            pred_res = (pred_energy - E_base) / N
            total_energy_loss += nn.MSELoss(reduction='sum')(pred_res, y_res).item()
            num_graphs += y_res.numel()

        # 3) Force loss
        if hasattr(batch, 'y_forces'):
            pred_forces = compute_forces(pred_energy, batch.pos)
            true_forces = batch.y_forces

            if hasattr(batch, "batch"):
                eq_node_mask = batch.is_eq[batch.batch]
            else:
                eq_node_mask = batch.is_eq.new_full((batch.pos.size(0),), bool(batch.is_eq.item()))
            neq_node_mask = ~eq_node_mask

            if neq_node_mask.any():
                total_force_loss += mse_loss_fn(pred_forces[neq_node_mask], true_forces[neq_node_mask]).item()
                num_atoms_neq += neq_node_mask.sum().item() * 3 # *3 for Fx,Fy,Fz

            if eq_node_mask.any():
                total_force_loss += huber_loss_fn(pred_forces[eq_node_mask], true_forces[eq_node_mask]).item()
                num_atoms_eq += eq_node_mask.sum().item() * 3 # *3 for Fx,Fy,Fz

    avg_energy_rmse = (total_energy_loss / num_graphs)**0.5 if num_graphs > 0 else 0.0
    avg_force_rmse = (total_force_loss / (num_atoms_neq + num_atoms_eq))**0.5 if (num_atoms_neq + num_atoms_eq) > 0 else 0.0
    
    print(f"\nEval ▶ Energy RMSE: {avg_energy_rmse:.4f} | Force RMSE: {avg_force_rmse:.4f}")
    
    # For compatibility with old reporting, we can return these values
    # The "total" loss is less meaningful now, so we focus on component RMSEs
    avg_e_loss_mse = total_energy_loss / num_graphs if num_graphs > 0 else 0.0
    avg_f_loss_mse = total_force_loss / (num_atoms_neq + num_atoms_eq) if (num_atoms_neq + num_atoms_eq) > 0 else 0.0

    return avg_e_loss_mse + avg_f_loss_mse, avg_e_loss_mse, avg_f_loss_mse


# Example usage and training script




# Training Started at 09:11am July 19
# Training with hard on build graph | RBF with Cosine | Hard Cutoff on Model Side