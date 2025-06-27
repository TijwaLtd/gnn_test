import torch

class GaussianRBF(torch.nn.Module):
    def __init__(self, num_basis: int, cutoff: float):
        super().__init__()
        # create evenly-spaced Gaussian centers on [0, cutoff]
        centers = torch.linspace(0.0, cutoff, num_basis)
        self.register_buffer("centers", centers)               
        # width = spacing between centers
        self.scale = (centers[1] - centers[0]).item()          

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        r: Tensor of shape [E] distances
        returns: [E, num_basis] Gaussian RBF features
        """
        # compute |r - center| for each basis
        diff = r.unsqueeze(1) - self.centers.unsqueeze(0)  # [E, num_basis]
        return torch.exp(-0.5 * (diff / self.scale) ** 2)