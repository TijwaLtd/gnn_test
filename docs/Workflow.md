# Project Workflow and Summary of Changes

This document provides a summary of the recent improvements and a step-by-step guide on how to run the project from scratch.

## Summary of Implemented Changes

The project has been significantly refactored and improved to address the challenge of accurately predicting forces at equilibrium. The following key changes were made:

### 1. Data-Centric Improvements
*   **Data Augmentation**: A `Jitter` transform is now applied during training, which adds small random noise to the positions of equilibrium structures. This teaches the model to learn a flat, stable energy minimum, discouraging the prediction of spurious forces.
*   **Balanced Sampling**: A `WeightedRandomSampler` has been implemented for the training data. This over-samples the minority class (equilibrium structures), ensuring the model pays sufficient attention to them during training.

### 2. Hybrid Loss Function
*   The force loss has been upgraded to a hybrid model. It now uses the standard `MSELoss` for non-equilibrium structures but switches to a more forgiving `HuberLoss` for equilibrium structures. This prevents the model from being harshly penalized for small, insignificant deviations from a perfect zero force.

### 3. Advanced Training Strategy
*   **Progressive Force Weighting**: The weight of the force component in the loss function (`lambda_forces`) is no longer fixed. It now starts small and linearly increases over a set number of "warmup" epochs. This allows the model to first learn a stable overall energy surface before progressively refining the force predictions.

### 4. Improved Workflow & Structure
*   **Code Restructuring**: The entire project has been reorganized into a clean `src` directory with modules for `data`, `processing`, `models`, `training`, and `evaluation`. All old, unused, and documentation files have been moved into `unused` and `docs` folders respectively, cleaning up the root directory.
*   **Streamlined Data Prep**: The multiple data preparation scripts have been consolidated into a single, easy-to-use script: `src/processing/prepare_splits.py`.
*   **Flexible Training**: The number of training epochs is no longer hardcoded. It can be easily set via the `--epochs` command-line argument.
*   **Cleaner Model Saving**: The model saving strategy has been simplified. The script now saves only `best_model.pth` (for the best-performing model) and `latest_checkpoint.pt` (for resuming training), preventing directory clutter.
*   **Enhanced Evaluation**: The evaluation script `src/evaluation/test_parity_2.py` now generates three separate parity plots: one for all data, one specifically for equilibrium structures (`forces_parity_EQ.png`), and one for non-equilibrium structures. This provides a clear and direct way to verify the fix for the equilibrium noise.
*   **Robust Data Handling**: The model's `forward` method and evaluation scripts now consistently use `torch_geometric.loader.DataLoader` for efficient batch processing of graph data, ensuring consistent data presentation and improved performance. The model's internal calculations for periodic boundary conditions have been refined to correctly handle lattice information.

---

## How to Run the Project From Scratch

Here is the end-to-end workflow for preparing the data, training the model, and evaluating the results.

### Step 1: Setup

First, set up your Python environment.

```bash
# Create a virtual environment
python -m venv venv

# Activate it (on macOS/Linux)
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

### Step 2: Prepare Data Splits

Ensure you have your non-equilibrium and equilibrium index files (e.g., `index_neq.csv`, `index_eq.csv`) in the `data/` directory.

Run the new preparation script to generate the `index_train.csv` and `index_val.csv` files that the training script needs.

```bash
python src/processing/prepare_splits.py --neq_csv data/index_neq.csv --eq_csv data/index_eq.csv
```

### Step 3: Train the Model

Run the training script. You can specify the number of epochs you want to train for.

```bash
# Train for the default 200 epochs
python src/training/train.py

# Or, train for a custom number of epochs (e.g., 150)
python src/training/train.py --epochs 150
```

The script will save `best_model.pth` and `latest_checkpoint.pt` in the `models/` directory.

### Step 4: Evaluate the Model

After training is complete, you can evaluate your best model using the `src/evaluation/test_parity_2.py` script. The script has several command-line options to control the evaluation process.

1.  **Default Evaluation:**
    By default, the script generates a single aggregate parity plot for all test samples.

    ```bash
    python src/evaluation/test_parity_2.py
    ```
    *   **Output:** `parity_plots/forces_parity_ALL.png`

2.  **Generating All Plots:**
    To generate a comprehensive set of plots (including per-sample, equilibrium-only, and non-equilibrium-only plots), use the `--gen-all-plots` flag.

    ```bash
    python src/evaluation/test_parity_2.py --gen-all-plots
    ```
    *   **Outputs:**
        *   `parity_plots/forces_parity_ALL.png`
        *   `parity_plots/forces_parity_EQ.png`
        *   `parity_plots/forces_parity_NEQ.png`
        *   Individual plots for each sample in the test set.

3.  **Handling Memory Errors:**
    If you encounter a `torch.cuda.OutOfMemoryError` (especially on systems prone to memory fragmentation), you can use the `--use-expandable-segments` flag. This flag can be combined with other flags.

    ```bash
    # Example: Generate all plots while using the expandable segments memory configuration
    python src/evaluation/test_parity_2.py --gen-all-plots --use-expandable-segments
    ```

4.  **Check the results:** The output plots will be saved in the `parity_plots/` directory. Pay special attention to `forces_parity_EQ.png` (when generated) to confirm that the noise at equilibrium has been reduced.

---

## Training on Google Colab

Training on a GPU in Google Colab can significantly speed up the process. Here’s how you can do it:

1.  **Open Google Colab:**
    *   Go to [colab.research.google.com](https://colab.research.google.com).
    *   Create a new notebook.

2.  **Enable GPU:**
    *   In the notebook, go to `Runtime` -> `Change runtime type`.
    *   Select `T4 GPU` from the "Hardware accelerator" dropdown and click `Save`.

3.  **Clone Your Project:**
    *   You will need to host your project on a Git repository (like GitHub, GitLab, etc.).
    *   In a Colab cell, clone your repository. You'll need to use a Personal Access Token (PAT) if the repository is private.

    ```python
    # For a public repository
    !git clone https://github.com/your-username/your-repository-name.git
    %cd your-repository-name

    # For a private repository
    # Replace <YOUR_PAT> with your Personal Access Token
    # Replace <YOUR_USERNAME> and <YOUR_REPO_NAME>
    !git clone https://<YOUR_PAT>@github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.git
    %cd <YOUR_REPO_NAME>
    ```

4.  **Upload Your Data:**
    *   The `data/` directory containing `DFT_DATA` and your index CSVs is too large to be included directly in the Git repository. You will need to upload it to your Google Drive.
    *   **Zip your `data` directory:** Create a `data.zip` file on your local machine.
    *   **Upload to Google Drive:** Upload `data.zip` to the root of your Google Drive.
    *   **Mount Google Drive in Colab:** In a new cell, mount your drive.

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
    *   **Unzip the data:** Unzip the data into your Colab project directory.

    ```python
    !unzip /content/drive/MyDrive/data.zip -d .
    ```

5.  **Install Dependencies:**
    *   In a new cell, install the required packages from your `requirements.txt` file.

    ```python
    !pip install -r requirements.txt
    ```

6.  **Run the Workflow:**
    *   Now you can follow the standard workflow steps within the Colab notebook cells.

    ```python
    # Step 1: Prepare Data Splits (if needed)
    !python src/processing/prepare_splits.py --neq_csv data/index_neq.csv --eq_csv data/index_eq.csv

    # Step 2: Train the Model
    # You can train for more epochs now that you have a GPU
    !python src/training/train.py --epochs 300

    # Step 3: Evaluate the Model
    # Remember to update the model path in the script if necessary
    !python src/evaluation/test_parity_2.py
    ```

7.  **Download Results:**
    *   Your results (plots in `parity_plots/`, logs in `logs/`, and trained models in `models/`) will be saved within the Colab environment. You can download them from the file browser on the left-hand side.
    *   For persistent storage, you can also save them directly to your mounted Google Drive by modifying the output paths in the scripts.