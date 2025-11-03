
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
        batch = data.batch  # [num_nodes]
        
        # — use precomputed PBC edges and shifts from the Data object —
        edge_index, edge_shift = data.edge_index, data.edge_shift      # [2,E], [E,3]
        lattice = data.lattice                                          # [3,3]

        source, target = edge_index
        # reconstruct the true Cartesian edge vectors under PBC
        edge_vec = (pos[target] + (edge_shift @ lattice)) - pos[source] # [E,3]
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

def train_one_epoch(model, loader, loss_fn, optimizer, device, lambda_forces=1, lambda_eq = 0.5, tau = 0.1):
    """Training loop with both energy and force losses, using per-atom residuals."""
    model.train()
    total_loss = 0.0
    energy_loss_total = 0.0
    force_loss_total = 0.0
    eq_penalty_total = 0.0
    
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        batch.pos.requires_grad_(True)
        
        try:
            # Predict total energy and forces
            pred_energy = model(batch)       # [batch_size]
            pred_forces = compute_forces(pred_energy, batch.pos)

            # --- Residual-based energy loss ---
            # Counts per graph (batch_size assumed 1)
            z = batch.z                                 # [num_nodes]
            nB = (z == 5).sum().float()                # scalar
            nC = (z == 6).sum().float()                # scalar
            n_atoms = batch.n_atoms.float()            # scalar

            # Baseline energy: B, C contributions + global per-atom offset
            E_base = nB * E_ref_B + nC * E_ref_C + n_atoms * E0_per_atom

            # True per-atom residual target
            y_res = batch.y_energy_res                # scalar
            # Model prediction residual
            pred_res = (pred_energy - E_base) / n_atoms

            # Compute residual energy loss
            energy_loss = loss_fn(pred_res, y_res)
            # ------------------------------------

            # Force loss - Supervised 
            if hasattr(batch, 'y_forces'):
                force_loss = loss_fn(pred_forces, batch.y_forces)
            else:
                force_loss = torch.tensor(0.0, device=device)
            
            # EQ Penalty (dead-zone)
            # Ensure batch.is_eq exists (graph-level, shape [num_graphs])
            if not hasattr(batch, "is_eq"):
                # default: treat as NEQ if dataset didn’t supply is_eq
                batch.is_eq = torch.zeros(getattr(batch, "num_graphs", 1), dtype=torch.bool, device=device)

            # Lift to node level for masking forces
            if hasattr(batch, "batch"):
                eq_node_mask = batch.is_eq[batch.batch]          # [num_nodes] bool
            else:
                # single-graph case, repeat for all its nodes
                eq_node_mask = batch.is_eq.new_full((batch.pos.size(0),), bool(batch.is_eq.item()))
            
            loss_eq = F.relu(pred_forces[eq_node_mask].abs() - tau).mean() if eq_node_mask.any() \
                    else torch.tensor(0.0, device=device)
            
            # Total batch loss
            total_batch_loss = energy_loss + lambda_forces * force_loss + lambda_eq * loss_eq

            # Backprop and optimization
            optimizer.zero_grad()
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate losses
            batch_size = getattr(batch, "num_graphs", 1)
            total_loss += total_batch_loss.item() * batch_size
            energy_loss_total += energy_loss.item() * batch_size
            force_loss_total += force_loss.item() * batch_size
            eq_penalty_total += loss_eq.item() * batch_size

        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    # Compute averages
    num_samples = len(loader.dataset) if hasattr(loader, 'dataset') else len(loader)
    avg_total_loss = total_loss / num_samples
    avg_energy_loss = energy_loss_total / num_samples
    avg_force_loss = force_loss_total / num_samples
    avg_eq_loss = eq_penalty_total / num_samples
    
    return avg_total_loss, avg_energy_loss, avg_force_loss, avg_eq_loss

def evaluate_model(model, loader, loss_fn, device, tau=0.1):
    """
    Evaluation loop computing:
      - avg_total_loss (residual‐MSE + force‐MSE)
      - avg_energy_loss (residual‐MSE)
      - avg_force_loss (force‐MSE)
      - prints per‐atom RMSE & MAE for energy residuals,
        and component‐wise RMSE & MAE for forces.
    """
    model.eval()
    total_loss = energy_loss_total = force_loss_total = 0.0

    # accumulators for metrics
    sum_sq_e = sum_abs_e = 0.0
    count_e = 0
    sum_sq_f = sum_abs_f = 0.0
    count_f = 0

    sum_abs_f_eq = 0.0; count_f_eq = 0
    below_tau_eq = 0

    for batch in loader:
        batch = batch.to(device)
        batch.pos.requires_grad_(True)

        # 1) forward pass (keep grad for forces)
        pred_energy = model(batch).view(-1)  # [batch_size]

        # 2) build residual‐energy target & prediction
        z      = batch.z
        nB     = (z == 5).sum().float()
        nC     = (z == 6).sum().float()
        N      = batch.n_atoms.float()
        E_base = nB * E_ref_B + nC * E_ref_C + N * E0_per_atom

        with torch.no_grad():
            # true per‐atom residual
            y_res    = batch.y_energy_res    # [batch_size]
            pred_res = (pred_energy - E_base) / N

            # energy MSE
            energy_loss = loss_fn(pred_res, y_res)

            # accumulate energy metrics
            err_e = (pred_res - y_res).item()
            sum_sq_e  += err_e**2
            sum_abs_e += abs(err_e)
            count_e   += 1

        # 3) force loss & metrics
        if hasattr(batch, 'y_forces'):
            pred_forces = compute_forces(pred_energy, batch.pos)
            force_loss  = loss_fn(pred_forces, batch.y_forces)

            diff = (pred_forces - batch.y_forces).detach().view(-1).cpu().numpy()
            sum_sq_f  += (diff**2).sum()
            sum_abs_f += np.abs(diff).sum()
            count_f   += diff.size
        else:
            force_loss = torch.tensor(0.0, device=device)
            pred_forces = compute_forces(pred_energy, batch.pos)

        if hasattr(batch, "is_eq"):
            if hasattr(batch, "batch"):
                eq_node_mask = batch.is_eq[batch.batch]
            else:
                eq_node_mask = batch.is_eq.new_full((batch.pos.size(0),), bool(batch.is_eq.item()))
            if eq_node_mask.any():
                f_eq = pred_forces[eq_node_mask].detach().norm(dim=1)  # per-atom norms
                sum_abs_f_eq += f_eq.abs().sum().item()
                count_f_eq   += f_eq.numel()
                below_tau_eq += (f_eq <= tau).sum().item()


        # 4) total loss tracking
        total_loss        += (energy_loss + force_loss).item()
        energy_loss_total += energy_loss.item()
        force_loss_total  += force_loss.item()

    # normalize by number of batches
    n_batches       = len(loader)
    avg_total_loss  = total_loss/ n_batches
    avg_energy_loss = energy_loss_total / n_batches
    avg_force_loss  = force_loss_total / n_batches

    # compute RMSE & MAE
    rmse_energy = (sum_sq_e  / count_e)**0.5 if count_e else 0.0
    mae_energy  = (sum_abs_e / count_e) if count_e else 0.0
    
    if count_f > 0:
        rmse_force = (sum_sq_f  / count_f)**0.5
        mae_force  = (sum_abs_f / count_f)
    else:
        rmse_force = mae_force = 0.0

    mean_abs_force_eq = (sum_abs_f_eq/ count_f_eq) if count_f_eq else 0.0
    pct_eq_below_tau = (100 * below_tau_eq / count_f_eq) if count_f_eq else 0.0

    print(f"\nEval ▶ Total Loss {avg_total_loss:.4f}  |  "
          f"E_MSE {avg_energy_loss:.4f}  |  F_MSE {avg_force_loss:.4f}")
    print(f"     ▶ Energy   RMSE {rmse_energy:.4f},  MAE {mae_energy:.4f}")
    print(f"     ▶ Forces   RMSE {rmse_force:.4f},  MAE {mae_force:.4f}\n")
    
    if count_f_eq:
        print(f"     ▶ EQ: mean |F| = {mean_abs_force_eq:.4e} eV/Å,  "
              f"% atoms ≤ τ({tau}) = {pct_eq_below_tau:.1f}%\n")   
    
    return avg_total_loss, avg_energy_loss, avg_force_loss


# Example usage and training script
if __name__ == "__main__":
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model with reasonable defaults
    model = E3NNForceModel(
        num_atom_types=120,          # Adjust based on your dataset
        cutoff=6.4,                  # Cutoff radius in Angstroms
        num_rbf=64,                  # Number of RBF basis functions
        max_l=3,                     # Maximum angular momentum for SH
        irreps_hidden="32x0e + 16x1o + 8x2e + 4x3o",  # Hidden irreps
        num_layers=3                 # Number of message passing layers # 2 for kalman
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.8e-4, weight_decay=2.4e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=8
    )
    loss_fn = nn.MSELoss()
    
    # Load your dataset
    from lazy_graph_dataset import LazyGraphDataset
    from torch_geometric.loader import DataLoader

    # Split 
    ROOT = r"D:/Sara/All_DFT_Data"
    train_ds = LazyGraphDataset(index_csv=r"C:\Users\labadmin\Documents\GNN\GNN-ML-Model\index_train.csv", root_dir = ROOT)
    val_ds = LazyGraphDataset(index_csv=r"C:\Users\labadmin\Documents\GNN\GNN-ML-Model\index_val.csv", root_dir= ROOT)
    
    # Data loaders
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=2)
    
    all_natoms = torch.tensor([ g.n_atoms for g in train_ds ], dtype=torch.float)
    avg_natoms = all_natoms.mean().item()      # a Python float
    lambda_forces = avg_natoms / 3.0           # dynamic force weight # HIGHER VALUE MEANS FORCE PRIORITIZE , LOWER VALUE MEANS ENERGY USE OPTUNA
    print(f"Using λ_force = {lambda_forces:.3f}")
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 20
    
        
        
    best_val = float('inf')
    lambda_eq = 0.6
    tau = 0.1
    for epoch in range(1, config["num_epochs"] + 1):
        train_loss, train_energy_loss, train_force_loss, train_eq_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, lambda_forces= lambda_forces, lambda_eq = lambda_eq, tau = tau)
        val_loss,   val_energy_loss, val_force_loss   = evaluate_model(model, val_loader, loss_fn, device, tau= tau)

        print(f"Train Metrics {epoch:03d} | "
          f"Train: total={train_loss:.4f} E={train_energy_loss:.4f} F={train_force_loss:.4f} EQpen={train_eq_loss:.4f} | "
          f"Val: total={val_loss:.4f} E={val_energy_loss:.4f} F={val_force_loss:.4f} | "
          f"λ_eq={lambda_eq:.2f}, τ={tau:.2f}")
        
        # — always append RMSEs for this epoch
        save_epoch_metrics(
            epoch,
            np.sqrt(train_energy_loss), np.sqrt(train_force_loss),
            np.sqrt(val_energy_loss),   np.sqrt(val_force_loss),
            filename='logs/epoch_metrics.out'
        )

        # — always append embeddings for this epoch
        save_embeddings_cumulative(
            model,
            filename='logs/embeddings.out',
            epoch=epoch
        )

        # — heavy dumps every 50 epochs or on the last epoch
        if epoch % 50 == 0 or epoch == config["num_epochs"]:
            save_energy_data(model, train_loader, f'logs/train_energy_epoch_{epoch}.out', device)
            save_energy_data(model, val_loader,   f'logs/val_energy_epoch_{epoch}.out',   device)
            save_force_data( model, train_loader, f'logs/train_force_epoch_{epoch}.out',  device)
            save_force_data( model, val_loader,   f'logs/val_force_epoch_{epoch}.out',    device)
            
            vstr = f"{float(val_loss):.6f}"

            torch.save(model.state_dict(),
               f"models/e3nn_weights_val_{vstr}_epoch_{epoch:04d}.pth")
            print(f"[ckpt] saved models/e3nn_weights_val_{vstr}_epoch_{epoch:04d}.pth")

            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': val_loss,
                'config': config,}
            torch.save(ckpt, f'models/epoch_{epoch:04d}_full.pt')
            print(f"[ckpt] saved models/epoch_{epoch:04d}_full.pt")
            
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        if epoch % 1 == 0:
            print(f"Epoch {epoch:3d} | "
                  f"Train: {train_loss:.6f} (E: {train_energy_loss:.6f}, F: {train_force_loss:.6f}), EQ: {train_eq_loss:.6f} | "
                  f"Val: {val_loss:.6f} (E: {val_energy_loss:.6f}, F: {val_force_loss:.6f})")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            vstr = f"{float(best_val_loss):.6f}"

            # full checkpoint (resume training)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': best_val_loss,
                'config': config,
            }, f"models/best_e3nn_model_val_{vstr}_epoch_{epoch:04d}.pt")

            # weights-only (inference)
            torch.save(model.state_dict(),
                    f"models/best_e3nn_weights_val_{vstr}_epoch_{epoch:04d}.pth")

            print(f"[ckpt] improved best → {best_val_loss:.6f}; "
                f"saved models/best_e3nn_model_val_{vstr}_epoch_{epoch:04d}.pt "
                f"+ best_e3nn_weights_val_{vstr}_epoch_{epoch:04d}.pth")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break



# Training Started at 09:11am July 19
# Training with hard on build graph | RBF with Cosine | Hard Cutoff on Model Side