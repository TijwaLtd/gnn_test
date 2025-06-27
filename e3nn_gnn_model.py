import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import radius_graph
from torch_scatter import scatter

import e3nn
from e3nn import o3
from e3nn.o3 import Irreps, FullyConnectedTensorProduct, Linear
from e3nn.nn import Gate, NormActivation
from e3nn.math import soft_one_hot_linspace


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
    def __init__(self, max_l=2):
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
        
        # MLP for processing RBF features
        # self.rbf_mlp = nn.Sequential(
            #nn.Linear(num_rbf, num_rbf),
            #nn.SiLU(),
            #nn.Linear(num_rbf, self.irreps_node_input.num_irreps)
        #)
        self.rbf_mlp = nn.Sequential(
            nn.Linear(num_rbf, num_rbf),
            nn.SiLU(),
            nn.Linear(num_rbf, self.irreps_node_input.dim)   # <-- .dim gives total channels
        )
        
        # Tensor product for message construction
        self.tp = FullyConnectedTensorProduct(
            self.irreps_node_input,
            self.irreps_edge_attr,
            self.irreps_node_hidden,
            shared_weights=False
        )
        
        tp_out = self.tp.irreps_out
        if tp_out != self.irreps_node_hidden:
            self.tp_proj = Linear(tp_out, self.irreps_node_hidden)
        else:
            self.tp_proj = nn.Identity()

        # Get the actual output dimension from tensor product
        tp_irreps = self.tp.irreps_out
        
        # Linear layer to combine tensor product outputs
        if tp_irreps != self.irreps_node_hidden:
            self.linear = Linear(tp_irreps, self.irreps_node_hidden)
        else:
            self.linear = nn.Identity()
        
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
        
        # Process RBF features to get weights for each irrep
        rbf_weights = self.rbf_mlp(rbf_features)  # [num_edges, num_irreps]
        
        # Construct messages via tensor product
        messages = self.tp_proj(node_features[source], edge_attr)  # [num_edges, tp_irreps]
        
        # Weight messages by RBF-derived weights
        # Apply weights by broadcasting across the irreps
        # weighted_messages = messages * rbf_weights.mean(dim=1, keepdim=True)
        weighted_messages = messages * rbf_weights
        
        # Linear transformation if needed
        messages = self.linear(weighted_messages)
        
        # Aggregate messages at target nodes
        num_nodes = node_features.size(0)
        aggregated_messages = scatter(messages, target, dim=0, dim_size=num_nodes, reduce='sum')
        
        # Add self-connection (residual)
        self_contribution = self.self_connection(node_features)
        
        return aggregated_messages + self_contribution

class ManualGatedBlock(nn.Module):
    """Gated nonlinearity: applies a linear projection followed by gate on non-scalars."""
    def __init__(self, input_irreps, output_irreps):
        super().__init__()
        self.lin = Linear(input_irreps, output_irreps)
        irreps = Irreps(output_irreps)
        # split scalars (l=0) and higher-order parts
        scalars = [(mul, ir) for mul, ir in irreps if ir.l == 0]
        non_scalars = [(mul, ir) for mul, ir in irreps if ir.l > 0]
        gate_irreps = Irreps(scalars) + Irreps(non_scalars)
        # one activation per scalar gate, one sigmoid per gated feature
        act_scalars = [nn.SiLU() for _ in Irreps(scalars)]
        act_gates = [torch.sigmoid for _ in Irreps(non_scalars)]
        self.gate = Gate(gate_irreps, act_scalars, act_gates)

    def forward(self, x):
        x = self.lin(x)
        return self.gate(x)
    
class E3NNForceModel(nn.Module):
    """Complete E3NN equivariant model for energy and force prediction"""
    def __init__(
        self,
        num_atom_types=100,
        cutoff=5.0,
        num_rbf=20,
        max_l=2,
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
            # Separate scalars and non-scalars for gating
            scalars = []
            gates = []
            nonscalars = []
            
            for mul, ir in self.irreps_hidden:
                if ir.l == 0 and ir.p == 1:  # even scalars
                    scalars.extend([ir] * mul)
                    gates.extend([ir] * mul)  # gates are also scalars
                elif ir.l > 0:
                    nonscalars.extend([ir] * mul)
            
            if len(gates) > 0 and len(nonscalars) > 0:
                # Create gate irreps: gates + gated features
                gate_irreps = o3.Irreps(gates) + o3.Irreps(nonscalars)
                act_scalars = [torch.sigmoid] * len(gates) + [None] * len(nonscalars)
                act_gates = [torch.sigmoid] * len(gates) + [torch.tanh] * len(nonscalars)
                
                self.gates.append(Gate(gate_irreps, act_scalars, act_gates))
            else:
                # If no gating needed, just use activation on scalars
                self.gates.append(NormActivation(self.irreps_hidden, scalar_nonlinearity=torch.silu))
        
        # Output head for energy (scalar prediction only)
        # Extract only scalar (l=0, p=1) features for energy prediction
        irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l == 0 and ir.p == 1])
        
        if len(irreps_scalars) > 0:
            self.energy_head = nn.Sequential(
                Linear(irreps_scalars, "64x0e"),
                NormActivation("64x0e", scalar_nonlinearity=torch.silu),
                Linear("64x0e", "32x0e"),
                NormActivation("32x0e", scalar_nonlinearity=torch.silu),
                Linear("32x0e", "1x0e")
            )
        else:
            # Fallback: project all features to scalars first
            self.energy_head = nn.Sequential(
                Linear(self.irreps_hidden, "64x0e"),
                NormActivation("64x0e", scalar_nonlinearity=torch.silu),
                Linear("64x0e", "1x0e")
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
        
        # Build edges if not provided
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            edge_index = data.edge_index
        else:
            edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        
        # Remove self-loops and filter by cutoff
        source, target = edge_index
        edge_vec = pos[target] - pos[source]  # [num_edges, 3]
        edge_length = torch.norm(edge_vec, dim=1)  # [num_edges]
        
        # Filter edges by cutoff and remove self-loops
        edge_mask = (edge_length < self.cutoff) & (edge_length > 1e-6)
        edge_index = edge_index[:, edge_mask]
        edge_vec = edge_vec[edge_mask]
        edge_length = edge_length[edge_mask]
        
        # Normalize edge vectors for spherical harmonics
        edge_vec_normalized = edge_vec / (edge_length.unsqueeze(1) + 1e-8)
        
        # Compute edge features
        rbf_features = self.rbf_expansion(edge_length)  # [num_edges, num_rbf]
        sh_features = self.spherical_harmonics(edge_vec_normalized)  # [num_edges, num_sh]
        
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

def train_one_epoch(model, loader, loss_fn, optimizer, device, lambda_forces=1.0):
    """Training loop with both energy and force losses"""
    model.train()
    total_loss = 0
    energy_loss_total = 0
    force_loss_total = 0
    
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        batch.pos.requires_grad_(True)  # Enable gradient computation for forces
        
        try:
            # Predict energy
            pred_energy = model(batch)
            
            # Compute energy loss
            if hasattr(batch, 'y_energy'):
                target_energy = batch.y_energy.view(-1)
            elif hasattr(batch, 'y'):
                target_energy = batch.y.view(-1)
            else:
                raise ValueError("No energy target found in batch (expected 'y_energy' or 'y')")
                
            energy_loss = loss_fn(pred_energy, target_energy)
            
            # Compute forces via gradient
            pred_forces = compute_forces(pred_energy, batch.pos)
            
            # Compute force loss
            if hasattr(batch, 'y_forces'):
                target_forces = batch.y_forces
                force_loss = loss_fn(pred_forces, target_forces)
            else:
                # If no force targets, set force loss to zero
                force_loss = torch.tensor(0.0, device=device)
            
            # Combined loss
            total_batch_loss = energy_loss + lambda_forces * force_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_batch_loss.backward()
            
            # Gradient clipping to prevent instability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate losses
            batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
            total_loss += total_batch_loss.item() * batch_size
            energy_loss_total += energy_loss.item() * batch_size
            force_loss_total += force_loss.item() * batch_size
            
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    num_samples = len(loader.dataset) if hasattr(loader, 'dataset') else len(loader)
    avg_total_loss = total_loss / num_samples
    avg_energy_loss = energy_loss_total / num_samples
    avg_force_loss = force_loss_total / num_samples
    
    return avg_total_loss, avg_energy_loss, avg_force_loss

def evaluate_model(model, loader, loss_fn, device):
    """Evaluation loop"""
    model.eval()
    total_loss = 0
    energy_loss_total = 0
    force_loss_total = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch.pos.requires_grad_(True)  # Still need gradients for force computation
            
            # Predict energy
            pred_energy = model(batch)
            
            # Energy loss
            if hasattr(batch, 'y_energy'):
                target_energy = batch.y_energy.view(-1)
            else:
                target_energy = batch.y.view(-1)
            energy_loss = loss_fn(pred_energy, target_energy)
            
            # Force loss (if targets available)
            if hasattr(batch, 'y_forces'):
                pred_forces = compute_forces(pred_energy, batch.pos)
                force_loss = loss_fn(pred_forces, batch.y_forces)
            else:
                force_loss = torch.tensor(0.0, device=device)
            
            batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
            total_loss += (energy_loss + force_loss).item() * batch_size
            energy_loss_total += energy_loss.item() * batch_size
            force_loss_total += force_loss.item() * batch_size
    
    num_samples = len(loader.dataset) if hasattr(loader, 'dataset') else len(loader)
    return total_loss / num_samples, energy_loss_total / num_samples, force_loss_total / num_samples

# Example usage and training script
if __name__ == "__main__":
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model with reasonable defaults
    model = E3NNForceModel(
        num_atom_types=100,          # Adjust based on your dataset
        cutoff=5.0,                  # Cutoff radius in Angstroms
        num_rbf=20,                  # Number of RBF basis functions
        max_l=2,                     # Maximum angular momentum for SH
        irreps_hidden="32x0e + 16x1o + 8x2e",  # Hidden irreps
        num_layers=4                 # Number of message passing layers
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=10, verbose=True
    )
    loss_fn = nn.MSELoss()
    
    # Example training loop (uncomment when you have your dataset ready)
    # Load your dataset
    from ML import LazyGraphDataset  
    from torch_geometric.loader import DataLoader
    
    dataset = LazyGraphDataset(index_csv="/Users/muhammadzainasad/Documents/Documents - Muhammad’s MacBook Air/Research Internship/Code/DFT_data/DFT_data/index.csv", root_dir="/Users/muhammadzainasad/Documents/Documents - Muhammad’s MacBook Air/Research Internship/Code/DFT_data/DFT_data")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 20
    
    for epoch in range(1, 101):
        # Training
        train_loss, train_energy_loss, train_force_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, lambda_forces=1.0
        )
        
        # Validation
        val_loss, val_energy_loss, val_force_loss = evaluate_model(
            model, val_loader, loss_fn, device
        )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | "
                  f"Train: {train_loss:.6f} (E: {train_energy_loss:.6f}, F: {train_force_loss:.6f}) | "
                  f"Val: {val_loss:.6f} (E: {val_energy_loss:.6f}, F: {val_force_loss:.6f})")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_e3nn_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    print(f"Training completed. Best validation loss: {best_val_loss:.6f}")