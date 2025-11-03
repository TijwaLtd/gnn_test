import os
import shutil

def clean_and_rename_subfolders(parent_dir):
    # Get all immediate subdirectories
    subfolders = sorted([
        os.path.join(parent_dir, d)
        for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    ])

    for idx, folder in enumerate(subfolders, start=1):
        print(f"Processing folder: {folder}")

        # Remove all files except OUTCAR and CONTCAR
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            if os.path.isfile(fpath):
                if fname not in ["OUTCAR", "CONTCAR"]:
                    os.remove(fpath)
                    print(f"Deleted: {fpath}")

        # Rename CONTCAR to POSCAR
        contcar_path = os.path.join(folder, "CONTCAR")
        poscar_path = os.path.join(folder, "POSCAR")
        if os.path.exists(contcar_path):
            os.rename(contcar_path, poscar_path)
            print(f"Renamed CONTCAR to POSCAR in {folder}")

        # Rename subfolder to TestX
        new_name = f"Test{idx}"
        new_folder_path = os.path.join(parent_dir, new_name)
        os.rename(folder, new_folder_path)
        print(f"Renamed folder to: {new_folder_path}")

if __name__ == "__main__":
    parent_directory = "/mnt/d/boron_carbide/GNN-test/New_Tests/Data/Data_eq"  # <-- Change this!
    clean_and_rename_subfolders(parent_directory)
