#!/usr/bin/env python3
import numpy as np
import torch
import matplotlib.pyplot as plt

# Paste or import your tanhu_cutoff_torch here
def tanhu_cutoff_torch(dists: torch.Tensor, rc: float) -> torch.Tensor:
    x = (1.0 - dists/rc).clamp(min=0.0)
    return x.tanh().pow(3)

# Paste or import your RBFExpansion here
class RBFExpansion(torch.nn.Module):
    def __init__(self, num_rbf=8, cutoff=6.3):
        super().__init__()
        self.centers = torch.linspace(0.0, cutoff, num_rbf)
        self.widths  = torch.full((num_rbf,), cutoff/num_rbf)
        self.cutoff  = cutoff

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        # pure Gaussian RBF
        return torch.exp(-((distances.unsqueeze(1) - self.centers) / self.widths)**2)

def main():
    cutoff = 6.3
    num_rbf = 8
    rbfe = RBFExpansion(num_rbf=num_rbf, cutoff=cutoff)

    # distances to test
    d_np = np.linspace(0.0, cutoff * 1.2, 500)
    d = torch.from_numpy(d_np).float()

    # compute RBF features
    rbf_feats = rbfe(d)              # [500, num_rbf]
    # pick a few basis indices to plot
    channels = [0, num_rbf//2, num_rbf-1]

    # compute TANHU envelope
    w = tanhu_cutoff_torch(d, cutoff)  # [500]

    # combined
    combined = rbf_feats * w.unsqueeze(-1)

    # Plot
    plt.figure(figsize=(8,5))
    for c in channels:
        plt.plot(d_np, rbf_feats[:,c].numpy(), label=f"RBF ch {c}")
    plt.plot(d_np, w.numpy(),       'k--', linewidth=2, label="TANHU cutoff")
    for c in channels:
        plt.plot(d_np, combined[:,c].numpy(), '--', label=f"RBF×TANHU ch {c}")
    plt.axvline(cutoff, color='red', linestyle=':', linewidth=1)
    plt.text(cutoff, 1.0, f" cutoff = {cutoff} Å", color='red', va='bottom')
    plt.xlabel("Distance r (Å)")
    plt.ylabel("Feature value")
    plt.title("Gaussian RBF channels and TANHU envelope")
    plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.show()

    # Quick numeric checks
    print(f"At r=0:   TANHU={w[0].item():.3f}")
    mid_idx = int(0.5 * len(d_np))
    print(f"At r=cutoff/2: TANHU={w[mid_idx].item():.3f}")
    end_idx = -1
    print(f"At r=1.2×cutoff: TANHU={w[end_idx].item():.3f}")

if __name__ == "__main__":
    main()



