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