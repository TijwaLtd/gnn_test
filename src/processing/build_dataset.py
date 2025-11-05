import sys
import os
import argparse

# Add the parent directory of 'src' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from processing.DFT_processor_2_Zain import DFTProcessor
from processing.generate_index import generate_index_main
from processing.prepare_splits import create_splits

def main():
    # Define data paths
    neq_data_path = "D:\Sara\All_DFT_Data"
    eq_data_path = "D:\Sara\All_DFT_Data\Test_Data"
    
    # Define output paths for index files and splits
    index_neq_csv = "data/index_neq.csv"
    index_eq_csv = "data/index_eq.csv"
    out_train_csv = "data/index_train.csv"
    out_val_csv = "data/index_val.csv"

    print("--- Running DFT Processor for Non-Equilibrium Data ---")
    neq_processor = DFTProcessor(dft_data_path=neq_data_path)
    neq_processor.process_directory()

    print("--- Running DFT Processor for Equilibrium Data ---")
    eq_processor = DFTProcessor(dft_data_path=eq_data_path)
    eq_processor.process_directory()

    print("--- Generating Index for Non-Equilibrium Data ---")
    generate_index_main(
        root_dir=neq_data_path,
        output=index_neq_csv,
        exclude_first=["Test_Data"]
    )

    print("--- Generating Index for Equilibrium Data ---")
    generate_index_main(
        root_dir=eq_data_path,
        output=index_eq_csv,
        is_eq=1
    )

    print("--- Creating Train/Validation Splits ---")
    create_splits(
        neq_csv=index_neq_csv,
        eq_csv=index_eq_csv,
        out_train=out_train_csv,
        out_test=out_val_csv
    )

    print("--- Data Processing Pipeline Complete ---")

if __name__ == '__main__':
    main()