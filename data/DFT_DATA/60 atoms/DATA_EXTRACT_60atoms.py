import os
import glob

# Function to read lattice vectors and atom positions from a POSCAR file
def read_POSCAR(file_path):
    lattice_vectors = []
    atom_positions = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Extract lattice vectors
        for line in lines[2:5]:
            lattice_vectors.append([float(val) for val in line.split()])
        # Extract atom positions
        for line in lines[8:68]:
            atom_positions.append([float(val) for val in line.split()])
    return lattice_vectors, atom_positions

# Function to extract the final energy and forces from an OUTCAR file
def extract_OUTCAR(file_path):
    final_energy = None
    final_forces = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Find the last occurrence of the energy line
        for line in reversed(lines):
            if "free  energy   TOTEN" in line:
                final_energy = float(line.split()[-2])  # Extract the energy value
                break
        # Find the last occurrence of the line containing forces
        start_index = None
        for i, line in enumerate(reversed(lines)):
            if "POSITION                                       TOTAL-FORCE (eV/Angst)" in line:
                start_index = len(lines) - i + 1  # Skip the header and start from the next line
                break
        if start_index is not None:
            for line in lines[start_index:]:
                if not line.strip():  # Check for empty line to break the loop
                    break
                if not all(char == '-' for char in line.strip()):  # Check if the line contains only dashes
                    # Extract the last three columns of forces
                    values = line.split()[-3:]
                    forces = [float(val) for val in values]
                    final_forces.append(forces)
    return final_energy, final_forces


import os
from openpyxl import Workbook

# Conversion factors
angstrom_to_bohr = 1.88973
ev_to_ha = 1 / 27.211386

# Parent folder containing 20 folders
parent_folder = '/mnt/d/boron_carbide/DFT_Structures/All_Data_ML/60/B45C15'

# Create a new Excel workbook
wb = Workbook()
# Create a new sheet for the data
ws = wb.active
ws.title = "Data"

# Iterate over each folder
for folder_name in os.listdir(parent_folder):
    folder_path = os.path.join(parent_folder, folder_name)
    if not os.path.isdir(folder_path):
        continue  # Skip if it's not a directory

    # Check if POSCAR and OUTCAR files exist in the folder
    poscar_file = os.path.join(folder_path, 'POSCAR')
    outcar_file = os.path.join(folder_path, 'OUTCAR')
    if os.path.exists(poscar_file) and os.path.exists(outcar_file):
        lattice_vectors, atom_positions = read_POSCAR(poscar_file)
        final_energy, final_forces = extract_OUTCAR(outcar_file)

        # Add headers
        headers = ["begin", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
        ws.append(headers)

        # Add lattice vectors in Bohr
        for vector in lattice_vectors:
            bohr_vector = [val * angstrom_to_bohr for val in vector]
            ws.append(["lattice", *bohr_vector, "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""])

        # Add atom positions in Bohr
        for i, position in enumerate(atom_positions, start=1):
            bohr_position = [val * angstrom_to_bohr for val in position]
            # Determine the type of atom based on the atom index
            atom_type = "B" if i <= 45 else "C"
            #print(i)
            # Convert final forces to Ha/Bohr
            #print(final_forces)
            final_forces_ha_bohr = [[val * 0.019446 for val in forces] for forces in final_forces]
            #print(final_forces_ha_bohr)
            ws.append(["atom", *bohr_position, atom_type, 0, 0, *final_forces_ha_bohr[i-1], "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""])

        # Convert final energy to Ha
        final_energy_ha = final_energy * ev_to_ha

        # Add energy and charge
        ws.append(["energy", final_energy_ha, "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""])
        ws.append(["charge", "0.00", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""])

# Add end
        ws.append(["end", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""])

# Specify the path where you want to save the Excel file
output_file = '/mnt/d/boron_carbide/DFT_Structures/All_Data_ML/60/B45C15/output.xlsx'
# Save the workbook
wb.save(output_file)

print(f"Data saved to {output_file}")

