# Plan to Improve Equilibrium Force Prediction in GNN

This document outlines a comprehensive plan to address the issue of inaccurate force predictions for equilibrium structures in the GNN model. The plan is designed to be implemented sequentially, allowing for validation at each stage.

### 1. Data-Centric Improvements: Augmentation & Balanced Sampling
*   **Goal:** Address the data imbalance and teach the model to recognize a stable energy minimum.
*   **Actions:**
    *   **Augmentation:** Modify the data loading process to apply small, random displacements (jitter) to the atomic positions of equilibrium structures. The force labels for these augmented structures will be kept as near-zero.
    *   **Balanced Sampler:** Implement a `torch.utils.data.WeightedRandomSampler` in the PyTorch DataLoader. This will involve assigning a higher sampling weight to the augmented equilibrium structures, ensuring the model sees them more frequently during training.

### 2. Loss Function Refinement: Hybrid Loss
*   **Goal:** Prevent the model from being overly penalized for small, noisy deviations from perfect zero forces in equilibrium structures.
*   **Actions:**
    *   Implement a hybrid loss function that uses a mask to differentiate between equilibrium and non-equilibrium structures within a training batch.
    *   **Non-Equilibrium Structures:** Continue using the standard Mean Squared Error (MSE) loss.
    *   **Equilibrium Structures:** Use a more forgiving loss function, such as `HuberLoss` or `SmoothL1Loss`, which creates a low-penalty zone around zero.

### 3. Advanced Training Strategy: Progressive Force Weighting
*   **Goal:** Allow the model to learn a stable energy surface first, then gradually refine the force predictions.
*   **Actions:**
    *   Implement a learning rate schedule for the force loss weight (`lambda_forces`).
    *   The weight will start small and be increased gradually over the training epochs, following a linear or exponential schedule.

### 4. Code Restructuring and Verification
*   **Goal:** Improve code maintainability and rigorously validate the modeling improvements.
*   **Actions:**
    *   After implementing the modeling changes, refactor the codebase to create a more logical file structure (e.g., separate directories for `data`, `models`, `training`, and `evaluation`).
    *   Throughout the process, use `test_parity_2.py` to evaluate performance. This evaluation will be enhanced to report metrics specifically for equilibrium vs. non-equilibrium structures to clearly track progress.
