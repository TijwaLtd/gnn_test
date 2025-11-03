# Fixes and Workarounds

This document lists common issues and their solutions for this project.

---

## 1. `torch.cuda.OutOfMemoryError` on High-End GPUs

### Problem

When running the evaluation script `src/evaluation/test_parity_2.py`, a `torch.cuda.OutOfMemoryError` was encountered on some machines with high-end NVIDIA GPUs (e.g., 16GB Quadro), while the same script with the same batch size ran successfully on machines with less VRAM (e.g., 4GB).

The error occurred during the force calculation step (`torch.autograd.grad`), which is known to be memory-intensive.

### Root Cause

The investigation revealed that the issue was not the total amount of available VRAM, but rather **memory fragmentation**.

The error message from PyTorch hinted at this: `If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.`

Even with a large amount of free VRAM, if the memory is fragmented into small, non-contiguous blocks, PyTorch may not be able to find a large enough contiguous block to allocate for a large tensor, leading to an OOM error. This can be dependent on the specific GPU, driver version, and other running processes.

### Solution

To address the memory fragmentation issue, we implemented a solution to set the `PYTORCH_CUDA_ALLOC_CONF` environment variable to `expandable_segments:True`. This changes how PyTorch's CUDA memory allocator works, making it more robust to fragmentation.

To provide flexibility, this is controlled by a command-line flag. The script `src/evaluation/test_parity_2.py` was modified to check for a `--use-expandable-segments` flag at startup. If this flag is present, it sets the environment variable before `torch` is imported.

### Usage

To run the evaluation script with the memory fragmentation fix enabled, use the `--use-expandable-segments` flag:

```bash
python src/evaluation/test_parity_2.py --use-expandable-segments
```

This should prevent the `OutOfMemoryError` on systems that are prone to memory fragmentation.

---

## 2. Poor Model Performance (High Energy Error, Incorrect Force Predictions)

### Problem
The model was exhibiting poor performance, characterized by:
- A very high and unstable validation energy RMSE.
- A parity plot showing a horizontal line of predictions at `NN Force = 0`, indicating the model was failing to predict forces for many structures.
- An extremely low R² score, confirming the model was not learning the underlying physics.

### Root Cause
Analysis of the training logs (`logs/epoch_metrics.out`) revealed that the validation force RMSE was decreasing, but the validation energy RMSE was not. This indicated that the force component of the loss function was overpowering the energy component.

The weight for the force loss (`lambda_forces_end`) was calculated as `avg_natoms / 3.0`, which resulted in a very large value that caused the model to prioritize minimizing force error at all costs, even if it meant learning a physically incorrect energy landscape.

### Solution
The weight of the force loss was reduced to provide a more balanced training objective. The calculation for `lambda_forces_end` in `src/training/train.py` was changed.

**Before:**
```python
lambda_forces_end = avg_natoms / 3.0
```

**After:**
```python
lambda_forces_end = 1.0
```
This gives the energy and force components of the loss a more equal footing, encouraging the model to learn a more physically meaningful potential energy surface.

---

## 3. Incorrect Handling of Equilibrium Structures

### Problem
The model was not learning to correctly predict zero forces for equilibrium structures, as evidenced by the large vertical spread of points at `DFT Force = 0` in the parity plot.

### Root Cause
An investigation of the training script (`src/training/train.py`) revealed a discrepancy with the project's `PLAN.md`. The plan intended to apply "jitter" (small random displacements) to equilibrium structures as a form of data augmentation to help the model learn the shape of the energy minimum.

However, the implementation was doing the opposite: it was only jittering *non-equilibrium* structures.

### Solution
The logic in the `ConditionalJitter` class in `src/training/train.py` was corrected to apply jitter to equilibrium structures as originally planned.

**Before:**
```python
class ConditionalJitter:
    def __init__(self, jitter_transform):
        self.jitter_transform = jitter_transform

    def __call__(self, data):
        if not data.is_eq.item():  # Apply jitter only if not equilibrium
            return self.jitter_transform(data)
        return data
```

**After:**
```python
class ConditionalJitter:
    def __init__(self, jitter_transform):
        self.jitter_transform = jitter_transform

    def __call__(self, data):
        if data.is_eq.item():  # Apply jitter only if it IS an equilibrium structure
            return self.jitter_transform(data)
        return data
```
This change aligns the implementation with the data-centric improvement strategy outlined in the project plan.

---

## 4. Project Structure and Module Improvements

The project has been refactored into a more organized and maintainable structure. The core logic is now located in the `src` directory, which is divided into the following modules:

### `src/processing`
This module contains all scripts related to data preparation and preprocessing.
- **Improvements:** The data preparation workflow has been streamlined. Scripts that were previously scattered have been consolidated, and the process of generating training/validation splits from the raw DFT data is now handled by a single script, `prepare_splits.py`.

### `src/data`
This module handles the loading of the processed data for training and evaluation.
- **Improvements:** The `LazyGraphDataset` class allows for efficient loading of graph data without having to load the entire dataset into memory at once. This is crucial for working with large datasets. The `transforms.py` file contains the `Jitter` transformation used for data augmentation.

### `src/models`
This module defines the Graph Neural Network architecture.
- **Improvements:** The `e3nn_gnn_model.py` file contains the `E3NNForceModel`, which is an equivariant GNN that correctly handles the geometric nature of the data. The model definition is now cleaner and more modular. It also includes the logic for the hybrid loss function (`train_one_epoch` and `evaluate_model`), which is a key part of the improved training strategy.

### `src/training`
This module contains the main script for training the model.
- **Improvements:** The `train.py` script orchestrates the entire training process. It incorporates several advanced techniques that have been added to improve model performance:
    - **Data Augmentation:** Applies jitter to equilibrium structures.
    - **Weighted Sampling:** Balances the training data between equilibrium and non-equilibrium structures.
    - **Progressive Force Weighting:** Gradually increases the weight of the force loss during training.
    - **Flexible Arguments:** Allows for easy configuration of training parameters like the number of epochs via command-line arguments.

### `src/evaluation`
This module is responsible for evaluating the performance of the trained model.
- **Improvements:** The `test_parity_2.py` script has been enhanced to provide a more robust evaluation of the model.
    - **Detailed Plots:** It can generate separate parity plots for all data, equilibrium-only data, and non-equilibrium-only data.
    - **Flexible Control:** The generation of these plots is controlled by a command-line flag (`--gen-all-plots`) for flexibility.
    - **Memory Management:** The documentation (`docs/Workflow.md`) now includes instructions on how to set the `PYTORCH_CUDA_ALLOC_CONF` environment variable to handle potential CUDA memory fragmentation issues.