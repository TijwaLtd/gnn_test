
import os
import torch
from DFT_processor_2_Zain import DFTProcessor

def is_valid_vasp_run(folder):
    return os.path.exists(os.path.join(folder, "POSCAR")) and os.path.exists(os.path.join(folder, "OUTCAR"))

def run_bulk_export(root_path):
    for dirpath, dirnames, filenames in os.walk(root_path):
        if is_valid_vasp_run(dirpath):
            print(f"\nProcessing VASP folder: {dirpath}")
            try:
                proc = DFTProcessor(dirpath)
                graphs = proc.process_directory()
                if graphs:
                    print(f"OK Saved {len(graphs)} graphs in {dirpath}/graphs.pt")
            except Exception as e:
                print(f"ERR Failed to process {dirpath}: {e}")


def export_graphs_from_index_custom(processor, index_csv, output_path, overwrite_graphs=True):
    """
    external-use graph export for Optuna or batch processing.
    """
    import os
    import pandas as pd
    from tqdm import tqdm

    df = pd.read_csv(index_csv)
    os.makedirs(output_path, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Graph Export (Optuna)"):
        rel_path = row['graph_path']
        abs_path = os.path.join(processor.root_dir, rel_path)
        graph_path = os.path.join(output_path, rel_path + ".pt")
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)

        if os.path.exists(graph_path) and not overwrite_graphs:
            continue

        try:
            graph = processor.process_folder(abs_path)
            torch.save(graph, graph_path)
        except Exception as e:
            print(f"[WARN] Could not process {rel_path}: {e}")

if __name__ == "__main__":
    root_vasp_folder = "D:\Sara\All_DFT_Data\Test_Data" 
    run_bulk_export(root_vasp_folder)
