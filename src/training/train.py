import os
import sys
import torch
import numpy as np
import argparse
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
from torch_scatter import scatter

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.e3nn_gnn_model import E3NNForceModel, compute_forces, train_one_epoch, evaluate_model, save_epoch_metrics, save_embeddings_cumulative, save_energy_data, save_force_data, E_ref_B, E_ref_C, E0_per_atom
from src.data.lazy_graph_dataset import LazyGraphDataset
from src.data.transforms import Jitter


class ConditionalJitter:
    def __init__(self, jitter_transform):
        self.jitter_transform = jitter_transform

    def __call__(self, data):
        if data.is_eq.item():  # Apply jitter only if it IS an equilibrium structure
            return self.jitter_transform(data)
        return data

def calculate_adaptive_lambda(model, loader, device, num_batches=20):
    """
    Calculate an adaptive lambda by comparing the magnitudes of the energy and force losses
    on a sample of the training data.
    """
    model.eval()
    
    total_energy_loss = 0.0
    total_force_loss = 0.0
    batches_processed = 0

    mse_loss_fn = torch.nn.MSELoss()
    huber_loss_fn = torch.nn.HuberLoss(delta=0.1)

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        
        batch = batch.to(device)
        batch.pos.requires_grad_(True)

        # --- Forward pass and force calculation with gradients enabled ---
        pred_energy = model(batch)
        pred_forces = compute_forces(pred_energy, batch.pos)

        # --- Loss calculations can be done without tracking further gradients ---
        with torch.no_grad():
            pred_forces = pred_forces.detach()

            # --- Energy Loss (Corrected) ---
            atom2graph = batch.batch
            n_graphs = pred_energy.size(0)
            
            ones = torch.ones_like(atom2graph, dtype=torch.float)
            n_atoms_per_graph = scatter(ones, atom2graph, dim=0, dim_size=n_graphs)

            isB = (batch.z == 5).float()
            isC = (batch.z == 6).float()
            nB_per_graph = scatter(isB, atom2graph, dim=0, dim_size=n_graphs)
            nC_per_graph = scatter(isC, atom2graph, dim=0, dim_size=n_graphs)

            E_base = nB_per_graph * E_ref_B + nC_per_graph * E_ref_C + n_atoms_per_graph * E0_per_atom
            
            y_res = batch.y_energy_res
            n_atoms_safe = torch.clamp(n_atoms_per_graph, min=1)
            pred_res = (pred_energy.detach() - E_base) / n_atoms_safe
            energy_loss = mse_loss_fn(pred_res, y_res)

            # --- Force Loss ---
            force_loss = torch.tensor(0.0, device=device)
            if hasattr(batch, 'y_forces'):
                true_forces = batch.y_forces
                if hasattr(batch, "batch"):
                    eq_node_mask = batch.is_eq[batch.batch]
                else:
                    eq_node_mask = batch.is_eq.new_full((batch.pos.size(0),), bool(batch.is_eq.item()))
                neq_node_mask = ~eq_node_mask

                if neq_node_mask.any():
                    force_loss += mse_loss_fn(pred_forces[neq_node_mask], true_forces[neq_node_mask])
                if eq_node_mask.any():
                    force_loss += huber_loss_fn(pred_forces[eq_node_mask], true_forces[eq_node_mask])

            total_energy_loss += energy_loss.item()
            total_force_loss += force_loss.item()
            batches_processed += 1

    model.train()

    if total_force_loss < 1e-9:
        return 1.0

    adaptive_lambda = total_energy_loss / total_force_loss
    return np.clip(adaptive_lambda, 0.1, 100.0)


def main():
    parser = argparse.ArgumentParser(description="Train an E3NN force model.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--dump_interval", type=int, default=50, help="How often (in epochs) to do heavy energy/force dumps.")
    parser.add_argument("--limit_samples", type=int, default=None, help="Limit the number of samples for quick testing.")
    parser.add_argument("--lambda_warmup_epochs", type=int, default=100, help="Number of epochs for force weighting warmup.")
    args = parser.parse_args()

    # Configuration
    config = {
        "num_epochs": args.epochs,
        "dump_interval": args.dump_interval,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training for {config['num_epochs']} epochs.")
    
    # Initialize model with reasonable defaults
    model = E3NNForceModel(
        num_atom_types=120,
        cutoff=6.4,
        num_rbf=64,
        max_l=3,
        irreps_hidden="32x0e + 16x1o + 8x2e + 4x3o",
        num_layers=3
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
    
    # Set the root directory to your DFT_DATA folder
    ROOT = "data/DFT_DATA"
    
    

    # --- Data Augmentation for Training Data ---
    jitter_transform = Jitter(displacement_magnitude=0.02)
    conditional_jitter_transform = ConditionalJitter(jitter_transform)

    # Create datasets
    train_ds = LazyGraphDataset(
        index_csv="data/index_train.csv",
        root_dir=ROOT,
        transform=conditional_jitter_transform,
        limit_samples=args.limit_samples
    )
    val_ds = LazyGraphDataset(
        index_csv="data/index_val.csv",
        root_dir=ROOT,
        limit_samples=args.limit_samples
    )
    
    # --- Weighted Sampler for Training Data ---
    is_eq_list = train_ds.df['is_eq'].values
    neq_count = (is_eq_list == 0).sum()
    eq_count = (is_eq_list == 1).sum()

    weight_neq = 1.0 / neq_count if neq_count > 0 else 0
    weight_eq = 1.0 / eq_count if eq_count > 0 else 0
    
    weights = [weight_eq if is_eq else weight_neq for is_eq in is_eq_list]
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)

    print(f"Sampler: {eq_count} EQ structures (weight={weight_eq:.4f}), "
          f"{neq_count} NEQ structures (weight={weight_neq:.4f})")

    # Data loaders
    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=os.cpu_count())
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=os.cpu_count())
    
    # --- Adaptive Force Weighting ---
    print(f"Using adaptive force weighting.")

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 20
    
    for epoch in range(1, config["num_epochs"] + 1):
        # Calculate adaptive lambda at the start of each epoch
        current_lambda_forces = calculate_adaptive_lambda(model, train_loader, device)
        print(f"Epoch {epoch:03d} | Calculated adaptive λ_force = {current_lambda_forces:.3f}")

        train_loss, train_energy_loss, train_force_loss, _ = train_one_epoch(model, train_loader, optimizer, device, lambda_forces=current_lambda_forces)
        val_loss, val_energy_loss, val_force_loss = evaluate_model(model, val_loader, device)

        print(f"Train Metrics {epoch:03d} | λ_F={current_lambda_forces:.3f} | "
              f"Train: total={train_loss:.4f} E={train_energy_loss:.4f} F={train_force_loss:.4f} | "
              f"Val: total={val_loss:.4f} E={val_energy_loss:.4f} F={val_force_loss:.4f}")
        
        save_epoch_metrics(
            epoch,
            np.sqrt(train_energy_loss), np.sqrt(train_force_loss),
            np.sqrt(val_energy_loss),   np.sqrt(val_force_loss),
            filename='logs/epoch_metrics.out'
        )

        save_embeddings_cumulative(
            model,
            filename='logs/embeddings.out',
            epoch=epoch
        )

        # --- Checkpoint Saving ---

        # Save latest checkpoint every 10 epochs for recovery
        if epoch % 10 == 0:
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': val_loss,
                'config': config,
            }
            torch.save(ckpt, 'models/latest_checkpoint.pt')
            print(f"[ckpt] Saved latest checkpoint at epoch {epoch}")

        # Save best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save weights only for inference
            torch.save(model.state_dict(), 'models/best_model.pth')
            
            # Save full checkpoint for resuming training from the best state
            best_ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': best_val_loss,
                'config': config,
            }
            torch.save(best_ckpt, 'models/best_model_full.pt')

            print(f"[ckpt] New best model found! Saved to models/best_model.pth (val_loss: {best_val_loss:.6f})")

        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Heavy data dumps (less frequent)
        if epoch % config["dump_interval"] == 0 or epoch == config["num_epochs"]:
            save_energy_data(model, train_loader, f'logs/train_energy_epoch_{epoch}.out', device)
            save_energy_data(model, val_loader,   f'logs/val_energy_epoch_{epoch}.out',   device)
            save_force_data( model, train_loader, f'logs/train_force_epoch_{epoch}.out',  device)
            save_force_data( model, val_loader,   f'logs/val_force_epoch_{epoch}.out',    device)
            print(f"[dump] Saved energy and force data at epoch {epoch}")

        scheduler.step(val_loss)

if __name__ == "__main__":
    main()
