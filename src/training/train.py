import os
import sys
import torch
import numpy as np
import argparse
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.e3nn_gnn_model import E3NNForceModel, compute_forces, train_one_epoch, evaluate_model, save_epoch_metrics, save_embeddings_cumulative, save_energy_data, save_force_data, E_ref_B, E_ref_C, E0_per_atom
from src.data.lazy_graph_dataset import LazyGraphDataset
from src.data.transforms import Jitter


class ConditionalJitter:
    def __init__(self, jitter_transform):
        self.jitter_transform = jitter_transform

    def __call__(self, data):
        if not data.is_eq.item():  # Apply jitter only if not equilibrium
            return self.jitter_transform(data)
        return data

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
    
    # --- Progressive Force Weighting ---
    all_natoms = torch.tensor([g.n_atoms for g in train_ds], dtype=torch.float)
    avg_natoms = all_natoms.mean().item()
    
    lambda_forces_start = 0.1
    lambda_forces_end = avg_natoms / 3.0
    lambda_warmup_epochs = args.lambda_warmup_epochs

    print(f"λ_force schedule: start={lambda_forces_start:.3f}, end={lambda_forces_end:.3f}, warmup={lambda_warmup_epochs} epochs")

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 20
    
    for epoch in range(1, config["num_epochs"] + 1):
        if epoch < lambda_warmup_epochs:
            current_lambda_forces = lambda_forces_start + (lambda_forces_end - lambda_forces_start) * (epoch / lambda_warmup_epochs)
        else:
            current_lambda_forces = lambda_forces_end

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
