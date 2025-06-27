
import os
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
                    print(f"✅ Saved {len(graphs)} graphs in {dirpath}/graphs.pt")
            except Exception as e:
                print(f"❌ Failed to process {dirpath}: {e}")

if __name__ == "__main__":
    root_vasp_folder = "/Users/muhammadzainasad/Documents/Documents - Muhammad’s MacBook Air/Research Internship/Code/ML/DFT_data/DFT_data" 
    run_bulk_export(root_vasp_folder)
