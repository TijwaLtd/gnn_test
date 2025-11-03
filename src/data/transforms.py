
import torch
from torch_geometric.data import Data

class Jitter(object):
    """
    Applies random displacement (jitter) to the atomic positions of a graph.
    This transform is applied only to equilibrium structures.
    """
    def __init__(self, displacement_magnitude=0.01):
        """
        Args:
            displacement_magnitude (float): The standard deviation of the Gaussian noise
                                            to be added to the positions.
        """
        self.displacement_magnitude = displacement_magnitude

    def __call__(self, data: Data) -> Data:
        """
        Applies the transform to the data object.

        Args:
            data (Data): A PyG Data object. It must have an `is_eq` attribute.

        Returns:
            Data: The transformed data object.
        """
        # Only apply jitter to equilibrium structures
        if hasattr(data, 'is_eq') and data.is_eq.item():
            noise = torch.randn_like(data.pos) * self.displacement_magnitude
            data.pos = data.pos + noise
        
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(displacement_magnitude={self.displacement_magnitude})'

