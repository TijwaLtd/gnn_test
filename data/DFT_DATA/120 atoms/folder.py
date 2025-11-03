import os

# Specify the path to the folder
folder_path = '/mnt/d/boron_carbide/DFT_Structures/All_Data_ML/120/B101C19'

# Get a list of all the folders (subfolders) in the specified folder
folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

# Print the names of the folders
for folder in folders:
    print(folder)
