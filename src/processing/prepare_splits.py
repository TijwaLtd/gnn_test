import argparse
import pandas as pd
from pathlib import Path

def create_splits(neq_csv, eq_csv, out_train, out_val, seed=42):
    """
    Creates training and validation splits from non-equilibrium and equilibrium data.
    """
    print("--- Creating Train/Val Splits ---")
    
    # Load data
    df_neq = pd.read_csv(neq_csv).copy()
    df_eq  = pd.read_csv(eq_csv).copy()

    # Split NEQ data
    neq_train = df_neq.sample(frac=0.8, random_state=seed)
    neq_val   = df_neq.drop(neq_train.index).copy()

    # Split EQ data
    eq_train = df_eq.sample(frac=0.8, random_state=seed)
    eq_val   = df_eq.drop(eq_train.index).copy()

    # Merge and shuffle
    train = pd.concat([neq_train, eq_train], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val   = pd.concat([neq_val,   eq_val],   ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Save to output files
    Path(out_train).parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_train, index=False)
    val.to_csv(out_val, index=False)

    print(f"Wrote train split to {out_train} ({len(train)} rows)")
    print(f"Wrote val split to {out_val} ({len(val)} rows)")
    print("--- Train/Val Splits Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train/validation splits for the GNN dataset.")
    parser.add_argument("--neq_csv",  required=True, help="CSV with all NEQ rows (e.g., data/index_neq.csv)")
    parser.add_argument("--eq_csv",   required=True, help="CSV with all EQ rows (e.g., data/index_eq.csv)")
    parser.add_argument("--out_train", default="data/index_train.csv", help="Output path for the training index.")
    parser.add_argument("--out_val",   default="data/index_val.csv", help="Output path for the validation index.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_splits(
        neq_csv=args.neq_csv,
        eq_csv=args.eq_csv,
        out_train=args.out_train,
        out_val=args.out_val,
        seed=args.seed
    )
