import argparse
import pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neq_csv",  required=True, help="CSV with all NEQ rows (e.g., index_neq.csv)")
    ap.add_argument("--eq_csv",   required=True, help="CSV with all EQ rows (e.g., index_eq_full.csv)")
    ap.add_argument("--neq_root", required=True, help="Root dir used for NEQ graph_path (e.g., D:/Sara/All_DFT_Data)")
    ap.add_argument("--eq_root",  required=True, help="Root dir used for EQ graph_path (e.g., D:/Sara/All_DFT_Data/Test_Data)")
    ap.add_argument("--out_train", required=True, help="Output: index_train.csv")
    ap.add_argument("--out_val",   required=True, help="Output: index_val.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Load
    df_neq = pd.read_csv(args.neq_csv).copy()
    df_eq  = pd.read_csv(args.eq_csv).copy()

    # Ensure flags + per-row root_dir exist
    df_neq["is_eq"] = 0
    df_eq["is_eq"]  = 1
    df_neq["root_dir"] = args.neq_root
    df_eq["root_dir"]  = args.eq_root

    # --- NEQ split: 80/20 ---
    neq_train = df_neq.sample(frac=0.8, random_state=args.seed)
    neq_val   = df_neq.drop(neq_train.index).copy()

    # --- EQ split: 80/20 ---
    # If not exactly 100 rows, frac keeps it proportional and robust.
    eq_train = df_eq.sample(frac=0.8, random_state=args.seed)
    eq_val   = df_eq.drop(eq_train.index).copy()

    # --- Merge into final train/val ---
    train = pd.concat([neq_train, eq_train], ignore_index=True)
    val   = pd.concat([neq_val,   eq_val],   ignore_index=True)

    # Optional: shuffle inside each split (keeps counts the same)
    train = train.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    val   = val.sample(frac=1.0,   random_state=args.seed).reset_index(drop=True)

    # Save
    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.out_train, index=False)
    val.to_csv(args.out_val, index=False)

    # Report
    print(f"[OK] wrote train → {args.out_train}  rows={len(train)} "
          f"(NEQ={int((train.is_eq==0).sum())}, EQ={int((train.is_eq==1).sum())})")
    print(f"[OK] wrote val   → {args.out_val}    rows={len(val)} "
          f"(NEQ={int((val.is_eq==0).sum())},   EQ={int((val.is_eq==1).sum())})")

if __name__ == "__main__":
    main()
