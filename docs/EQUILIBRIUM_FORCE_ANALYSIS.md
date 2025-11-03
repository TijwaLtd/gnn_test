# Analysis of Equilibrium Force Prediction in GNNs

This document outlines the reasons why Graph Neural Networks (GNNs) often struggle to accurately predict forces for atomic structures at or near equilibrium.

## 1. The Core of the Problem: Energy vs. Forces

The fundamental approach of this model, and many like it, is to predict the potential energy (a scalar value) of a given atomic structure. The forces (vector values) are not predicted directly; instead, they are derived as the analytical negative gradient of the predicted energy with respect to the atomic positions:

**F = -∇E**

This relationship is the primary source of the challenge. For the predicted forces to be zero, the learned potential energy surface must be perfectly **flat** at the equilibrium geometry.

## 2. The Challenge Illustrated

The provided parity plot (`image-2.png`) clearly illustrates this issue.

![Force Parity Plot](image-2.png)

- **Overall Accuracy:** The plot shows a high R² value (0.9837) and a low Mean Absolute Error (MAE), indicating that the model is very accurate for non-equilibrium structures with significant forces. The points lie tightly along the y=x line.
- **Equilibrium Error:** However, there is a large, dense cluster of points where the true DFT Force is near zero (the y-axis), but the predicted NN Force varies significantly, from approximately -2.5 to +2.5 eV/Å.

This "blob" at the center of the plot shows that even when the true forces are zero, the model often predicts spurious, non-zero forces.

## 3. Why Does This Happen?

### a) Data Distribution Imbalance

As noted in `Requirements from fiverr.md`, the training dataset is dominated by non-equilibrium structures. These structures have high potential energies and large associated forces. The model's loss function is therefore optimized primarily to be accurate in these high-gradient regions of the energy landscape. The relatively few equilibrium structures have a smaller impact on the overall training loss, so the model is not sufficiently incentivized to learn their specific characteristics.

### b) The Gradient Problem

For the model to predict a force of zero, the gradient of its learned energy function must be zero. Any small error, ripple, or imperfection in the learned energy surface around an equilibrium point will result in a non-zero gradient, leading to the spurious forces we see in the plot. It is extremely difficult for a neural network to learn a perfectly flat potential energy minimum based on a limited number of examples.

### c) The "Dead-Zone" Penalty (`lambda_eq` and `tau`)

The current implementation in `e3nn_gnn_model.py` correctly attempts to mitigate this with a "dead-zone" penalty for equilibrium structures:

`loss_eq = F.relu(pred_forces[eq_node_mask].abs() - tau).mean()`

This is an intelligent approach. It tells the model: "For equilibrium structures, I will only penalize you if the force you predict has a magnitude greater than a small threshold `tau`." This stops the model from chasing perfect zero (which is difficult) and instead encourages it to predict forces that are simply "small enough." While this helps, it doesn't completely solve the underlying gradient problem, as the model can still produce forces within the `[-tau, tau]` dead-zone without penalty.

## 4. Conclusion & Path Forward

The difficulty in predicting zero forces is not a simple bug but a fundamental consequence of deriving forces from a learned energy function. The model is not explicitly trained to predict forces, but to predict energies, and the data distribution makes it difficult to learn the subtle flatness of the energy surface at equilibrium points.

The next steps in this project will involve refining the existing methods and exploring new techniques to further incentivize the model to produce a flatter energy surface for equilibrium structures, thereby reducing the spurious forces and tightening the central cluster in the parity plot.
