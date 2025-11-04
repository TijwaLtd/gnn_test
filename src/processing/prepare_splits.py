import argparse
import pandas as pd
from pathlib import Path

def create_splits(neq_csv, eq_csv, out_train, out_test, seed=42):
    """
    Creates training and test splits from non-equilibrium and equilibrium data.
    Uses 80% for training and 20% for testing, with balanced representation of different data sources.
    """
    print("--- Creating Balanced Train/Test Splits (80/20) ---")
    
    # Load data with headers
    df_neq = pd.read_csv(neq_csv)
    df_eq = pd.read_csv(eq_csv)
    
    # Ensure we have the expected columns
    if 'root_dir' not in df_neq.columns or 'graph_path' not in df_neq.columns:
        raise ValueError("Input CSV files must have 'root_dir' and 'graph_path' columns")
    
    # Combine the full path
    df_neq['full_path'] = df_neq['root_dir'] + '/' + df_neq['graph_path']
    df_eq['full_path'] = df_eq['root_dir'] + '/' + df_eq['graph_path']
    
    # Add source information
    def get_source(path):
        if 'Test' in path:
            return 'test_data'
        elif 'DFT_DATA' in path or 'Data_eq' in path:
            return 'dft_data'
        return 'other'
    
    df_neq['source'] = df_neq['full_path'].apply(get_source)
    df_eq['source'] = df_eq['full_path'].apply(get_source)
    
    # Add a combined label for stratification
    for df in [df_neq, df_eq]:
        df['strata'] = df['source'] + '_' + df['is_eq'].astype(str)
    
    # Combine all data
    df_all = pd.concat([df_neq, df_eq], ignore_index=True)
    
    # Get unique strata for splitting
    unique_strata = df_all['strata'].unique()
    
    train_dfs = []
    test_dfs = []
    
    # Split each stratum separately
    for stratum in unique_strata:
        stratum_df = df_all[df_all['strata'] == stratum]
        if len(stratum_df) < 2:  # If only one sample, put in train
            train_dfs.append(stratum_df)
            continue
            
        # Use 80/20 split for this stratum
        train_stratum = stratum_df.sample(frac=0.8, random_state=seed)
        test_stratum = stratum_df.drop(train_stratum.index)
        
        train_dfs.append(train_stratum)
        test_dfs.append(test_stratum)
    
    # Combine all strata
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    # Shuffle the datasets
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    print("\n" + "="*50)
    print("SPLIT SUMMARY")
    print("="*50)
    print(f"Total samples: {len(df_all)}")
    print(f"  - Training set: {len(train_df)} samples ({len(train_df)/len(df_all):.1%})")
    print(f"  - Test set:     {len(test_df)} samples ({len(test_df)/len(df_all):.1%})")
    
    # Print stratification before dropping columns
    def print_stratification(df, name):
        print(f"\n{name} set ({len(df)} samples):")
        
        # Calculate source distribution
        source_counts = df['source'].value_counts()
        print("  By source:")
        for source, count in source_counts.items():
            print(f"    {source}: {count} samples ({count/len(df):.1%})")
        
        # Calculate equilibrium status distribution
        print("\n  By equilibrium status:")
        eq_counts = df['is_eq'].value_counts()
        for eq, count in eq_counts.items():
            print(f"    is_eq={eq}: {count} samples ({count/len(df):.1%})")
    
    print_stratification(train_df, "Training")
    print_stratification(test_df, "Test")
    
    # Prepare output columns, keeping the full path structure
    output_cols = ['root_dir', 'graph_path', 'is_eq', 'n_B', 'n_C', 'n_atoms', 'composition', 'tag']
    train_df = train_df[output_cols]
    test_df = test_df[output_cols]
    
    # Save to output files with headers
    Path(out_train).parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_train, index=False, header=True)
    test_df.to_csv(out_test, index=False, header=True)
    
    print(f"\n[OK] Wrote {len(train_df)} training samples to {out_train}")
    print(f"[OK] Wrote {len(test_df)} test samples to {out_test}")
    print("\n--- Train/Test Splits Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train/test splits (80/20) for the GNN dataset.")
    parser.add_argument("--neq_csv",  required=True, help="CSV with all NEQ rows (e.g., data/index_neq.csv)")
    parser.add_argument("--eq_csv",   required=True, help="CSV with all EQ rows (e.g., data/index_eq.csv)")
    parser.add_argument("--out_train", default="data/index_train.csv", help="Output path for the training index.")
    parser.add_argument("--out_test",  default="data/index_test.csv", help="Output path for the test index.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Ensure scikit-learn is available
    try:
        import sklearn
    except ImportError:
        print("Error: scikit-learn is required for stratified splitting. Please install it with:")
        print("pip install scikit-learn")
        sys.exit(1)

    create_splits(
        neq_csv=args.neq_csv,
        eq_csv=args.eq_csv,
        out_train=args.out_train,
        out_test=args.out_test,
        seed=args.seed
    )
