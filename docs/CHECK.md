# GNN Force Prediction Analysis & Recommendations

## Current Issues

1. **Force Prediction at Equilibrium**
   - The model struggles with predicting near-zero forces in equilibrium configurations
   - The error plot shows significant deviations from the y=x line around the zero-force region
   - This is a common issue when the training data lacks sufficient equilibrium examples

2. **Data Imbalance**
   - Non-equilibrium data (2200+ structures) dominates the training set
   - Only ~100 equilibrium structures are available, leading to poor generalization
   - The model is biased towards non-equilibrium force patterns

3. **Loss Function**
   - Current implementation uses a simple MSE loss for forces
   - The equilibrium penalty term (`loss_eq`) helps but may need refinement
   - No explicit handling of the different force magnitude regimes

## Recommended Improvements

### 1. Enhanced Data Strategy

```python
# 1. Data Augmentation for Equilibrium Configurations
def augment_equilibrium_data(original_data, num_augmentations=5):
    """Create synthetic equilibrium configurations by adding small displacements"""
    augmented_data = []
    for _ in range(num_augmentations):
        # Small random displacement (0.05-0.1Å)
        displacement = torch.randn_like(original_data.pos) * 0.05
        new_pos = original_data.pos + displacement
        
        # Create new data point with zero forces
        new_data = original_data.clone()
        new_data.pos = new_pos
        new_data.y_forces = torch.zeros_like(original_data.y_forces)
        new_data.is_eq = torch.tensor([True])
        augmented_data.append(new_data)
    return augmented_data

# 2. Dynamic Sampling
# Increase weight of equilibrium examples during training
class BalancedSampler(torch.utils.data.WeightedRandomSampler):
    def __init__(self, dataset, eq_weight=5.0):
        weights = torch.ones(len(dataset))
        for i, data in enumerate(dataset):
            if hasattr(data, 'is_eq') and data.is_eq.any():
                weights[i] = eq_weight
        super().__init__(weights, len(weights))
```

### 2. Improved Loss Function

```python
def force_loss(pred_forces, target_forces, eq_mask, tau=0.1):
    """
    Enhanced force loss with separate handling of equilibrium and non-equilibrium forces
    
    Args:
        pred_forces: [N, 3] predicted forces
        target_forces: [N, 3] target forces
        eq_mask: [N] boolean mask for equilibrium configurations
        tau: threshold for equilibrium forces (forces below tau are considered zero)
    """
    # Standard MSE for non-equilibrium forces
    neq_mask = ~eq_mask
    mse = F.mse_loss(pred_forces[neq_mask], target_forces[neq_mask], reduction='none')
    
    # Modified Huber loss for equilibrium forces (more forgiving near zero)
    diff = pred_forces[eq_mask]  # Should be zero for equilibrium
    huber = F.huber_loss(diff, torch.zeros_like(diff), delta=tau, reduction='none')
    
    # Combine losses
    return mse.mean() + 2.0 * huber.mean()  # Weight equilibrium more heavily
```

### 3. Model Architecture Modifications

1. **Force-Specific Regularization**
   ```python
   class E3NNForceModel(nn.Module):
       def __init__(self, ...):
           # ... existing initialization ...
           
           # Additional force-specific layers
           self.force_head = nn.Sequential(
               Linear(self.irreps_hidden, self.irreps_hidden),
               NormActivation(self.irreps_hidden, scalar_nonlinearity=torch.nn.functional.silu),
               Linear(self.irreps_hidden, o3.Irreps("1x1o"))  # Vector output for forces
           )
   
       def forward(self, data):
           # ... existing forward pass ...
           
           # Direct force prediction (alternative to autograd)
           forces = self.force_head(node_features).squeeze(-1)  # [num_nodes, 3]
           
           # Combine with autograd forces using a learned gate
           autograd_forces = compute_forces(energy, data.pos)
           gate = torch.sigmoid(self.force_gate(node_features).squeeze(-1))  # [num_nodes, 1]
           forces = gate * forces + (1 - gate) * autograd_forces
           
           return energy, forces
   ```

2. **Equilibrium-Aware Attention**
   ```python
   class EquilibriumAttention(nn.Module):
       def __init__(self, hidden_dim):
           super().__init__()
           self.query = nn.Linear(hidden_dim, hidden_dim)
           self.key = nn.Linear(hidden_dim, hidden_dim)
           self.value = nn.Linear(hidden_dim, hidden_dim)
           
       def forward(self, x, mask=None):
           Q = self.query(x)  # [N, hidden_dim]
           K = self.key(x)    # [N, hidden_dim]
           V = self.value(x)  # [N, hidden_dim]
           
           attention = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
           if mask is not None:
               attention = attention.masked_fill(mask == 0, -1e9)
           attention = F.softmax(attention, dim=-1)
           
           return torch.matmul(attention, V)  # [N, hidden_dim]
   ```

### 4. Training Strategy

1. **Curriculum Learning**
   - Start with a higher weight on equilibrium examples
   - Gradually increase the weight of non-equilibrium data
   - Implement learning rate warmup for stable training

2. **Progressive Training**
   ```python
   def get_lambda_schedule(epoch, max_epochs):
       """Gradually increase lambda_forces during training"""
       min_lambda = 0.1
       max_lambda = 1.0
       return min_lambda + (max_lambda - min_lambda) * (epoch / max_epochs)
   
   # In training loop
   current_lambda = get_lambda_schedule(epoch, max_epochs)
   loss = energy_loss + current_lambda * force_loss
   ```

## Implementation Plan

1. **Immediate Fixes (1-2 days)**
   - Implement the enhanced force loss function
   - Add data augmentation for equilibrium configurations
   - Adjust the equilibrium penalty weight (`lambda_eq`)

2. **Medium-term Improvements (3-5 days)**
   - Implement the force-specific model head
   - Add curriculum learning
   - Set up proper validation on a held-out equilibrium set

3. **Long-term Enhancements (1-2 weeks)**
   - Implement equilibrium-aware attention
   - Add more sophisticated data augmentation
   - Consider using a pretrained model for transfer learning

## Expected Outcomes

- **Short-term**: 20-30% reduction in force prediction errors around equilibrium
- **Medium-term**: Better generalization to unseen equilibrium configurations
- **Long-term**: State-of-the-art force prediction accuracy across all regimes

## Monitoring Progress

1. Track separate metrics for equilibrium and non-equilibrium examples
2. Plot force error distributions before and after implementation
3. Monitor the gate values in the force prediction head to understand model behavior

## Conclusion

The key to improving force prediction at equilibrium lies in better handling of the data imbalance and specialized architectural components for equilibrium configurations. The proposed changes should significantly improve the model's performance while maintaining its strengths in non-equilibrium prediction.
