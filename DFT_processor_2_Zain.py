import os
import numpy as np
import torch
from e3nn.o3 import Irreps, spherical_harmonics
from gaussian_rbf import GaussianRBF
from torch_geometric.data import Data
from openpyxl import Workbook

class DFTProcessor:
    def __init__(self, dft_data_path, lmax: int = 3, num_basis: int = 64, cutoff: float = 6.4): #cutoff = 6.2 thats what we were using with edgeconv..
        """Initialize DFT processor with path to data directory"""
        self.dft_data_path = os.path.abspath(dft_data_path)   # ✅ MUST come first
        self.root_dir = self.dft_data_path      

        self.graphs = []  # Store processed graphs

        #e3nn plumbing
        self.node_irreps = Irreps("1x0e") # 1x0e means 1 scalar and 0 vector
        self.edge_irreps = Irreps.spherical_harmonics(lmax)
        self.radial_model = GaussianRBF(num_basis=num_basis, cutoff=cutoff)
        self.cutoff = cutoff

    def __len__(self):
        """Make the processor lennable"""
        return len(self.graphs)
    
    def __getitem__(self, idx):
        """Make the processor indexable"""
        return self.graphs[idx]
        
    @staticmethod
    def read_POSCAR(file_path):
        """
        Reads a POSCAR file and returns the lattice vectors, atomic positions,
        and atom counts. This version applies the scaling factor properly.
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Line 2 contains the scaling factor
        scale = float(lines[1].strip())
        
        # Lines 3-5: Read lattice vectors and apply scaling
        lattice_vectors = np.array([[float(x) for x in line.split()] for line in lines[2:5]]) * scale
        
        # Line 7 contains atom counts (assuming format: "n_B n_C")
        atom_counts = lines[6].split()
        n_boron = int(atom_counts[0])
        n_carbon = int(atom_counts[1])
        element_to_z = {"B":5,"C":6}
        atom_types = [5]* n_boron+ [6]* n_carbon
        total_atoms = n_boron + n_carbon
    
        # Read atomic positions (assume they start at line 9)
        pos_start = 8
        positions = []
        for line in lines[pos_start:pos_start + total_atoms]:
            positions.append([float(x) for x in line.split()[:3]])
        atom_positions = np.array(positions)
        
        # Check if the positions are given in "Direct" (fractional) coordinates
        coord_type = lines[pos_start - 1].strip().lower()
        if "direct" in coord_type:
            # Convert fractional positions to Cartesian using lattice vectors
            atom_positions = np.dot(atom_positions, lattice_vectors)
        
        return lattice_vectors, atom_positions, n_boron, n_carbon, total_atoms

    def extract_OUTCAR(self, file_path):
        """Extract the final energy and forces from an OUTCAR file"""
        final_energy = None
        final_forces = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in reversed(lines):
                if "free  energy   TOTEN" in line:
                    final_energy = float(line.split()[-2])
                    break
            start_index = None
            for i, line in enumerate(reversed(lines)):
                if "POSITION                                       TOTAL-FORCE (eV/Angst)" in line:
                    start_index = len(lines) - i + 1
                    break
            
            if start_index is not None:
                for line in lines[start_index:]:
                    if not line.strip():
                        break
                    # Skip separator lines and total drift line
                    if (not all(char == '-' for char in line.strip()) and 
                        'total drift' not in line.lower() and
                        line.strip() and
                        not line.strip().startswith('--')):
                        try:
                            values = line.split()[-3:]
                            if len(values) == 3:  # Ensure we have fx, fy, fz
                                forces = [float(val) for val in values]
                                final_forces.append(forces)
                        except (ValueError, IndexError):
                            # Skip malformed force lines
                            continue
        return final_energy, final_forces

    def build_graph(
        self,
        atom_positions,
        atom_types,
        forces,
        energy,
        lattice_vectors,
        **_,
    ):
        """
        Returns a torch_geometric Data object with
        • z            : atomic numbers                     (node feature, 0e)
        • pos          : Cartesian positions                (shape [N,3])
        • edge_index   : [2, E] indices
        • edge_attr    : dict {rbf: [E,B], sh: [E,dim_sh]}  (edge features)
        • y_energy     : [1] total energy  (0e target)
        • y_forces     : [N,3] atomic forces (1o target)
        """
        positions = np.array(atom_positions)              # (N,3)
        n_atoms   = len(positions)

        # -------------------------------------------------
        # 1) build neighbour list
        # -------------------------------------------------
        offsets = [np.array([i, j, k])
                for i in (-1, 0, 1)
                for j in (-1, 0, 1)
                for k in (-1, 0, 1)]
        edge_list, edge_vecs, edge_lengths, edge_shifts = [], [], [], []

        edge_list, edge_vecs, edge_lengths = [], [], []
        for i in range(n_atoms):
            for j in range(n_atoms):
                for off in offsets:
                    if i == j and np.all(off == 0):
                        continue
                    img_pos = positions[j] + off.dot(lattice_vectors)
                    diff    = positions[i] - img_pos
                    dist    = np.linalg.norm(diff)
                    if dist < self.cutoff:
                        edge_list.append([i, j])
                        edge_vecs.append(diff)
                        edge_lengths.append(dist)
                        edge_shifts.append(off)

        if not edge_list:
            print("No edges within cutoff")
            return None

        edge_index   = torch.tensor(edge_list,    dtype=torch.long).t().contiguous()
        edge_vecs    = torch.tensor(edge_vecs,    dtype=torch.float)          # [E,3]
        edge_lengths = torch.tensor(edge_lengths, dtype=torch.float)          # [E]
        edge_shifts  = torch.tensor(edge_shifts, dtype=torch.float)            # [E,3]

        # -------------------------------------------------
        # 2) radial + angular edge attributes
        # -------------------------------------------------
        rbf = self.radial_model(edge_lengths)                                  # [E,B]
        sh  = spherical_harmonics(self.edge_irreps, edge_vecs, normalize=True)          # [E,dim_sh]

        edge_attr = {"rbf": rbf, "sh": sh}

        # -------------------------------------------------
        # 3) node / target tensors
        # -------------------------------------------------
        z        = torch.tensor([5 if t == "B" else 6 for t in atom_types], dtype=torch.long)
        pos      = torch.tensor(positions, dtype=torch.float)                 # [N,3]
        y_energy = torch.tensor([energy], dtype=torch.float)                  # [1]
        y_forces = torch.tensor(forces, dtype=torch.float)                    # [N,3]

        # -------------------------------------------------
        # 4) assemble Data object
        # -------------------------------------------------
        data = Data(
            z=z,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y_energy=y_energy,
            y_forces=y_forces,
            edge_shift=edge_shifts,                         # <<< NEW

        )

        # stash irreps for downstream convenience
        data.node_irreps = self.node_irreps
        data.edge_irreps = {"rbf": f"{rbf.shape[1]}x0e", "sh": self.edge_irreps}

        # optional extras
        data.n_atoms = torch.tensor(n_atoms, dtype=torch.long)

        data.lattice  = torch.tensor(lattice_vectors, dtype=torch.float)
        data.raw_energy = y_energy.clone()

        # load your fitted refs
        E_ref_B     = -0.809536
        E_ref_C     = -4.607478
        E0_per_atom = -5.417014

        z     = data.z
        nB    = (z == 5).sum().item()
        nC    = (z == 6).sum().item()
        n     = data.n_atoms.item()
        # total baseline energy
        E_base = nB*E_ref_B + nC*E_ref_C + n*E0_per_atom
        data.y_energy_res = (data.raw_energy.item() - E_base) / n


        return data
        
       

    def process_directory(self):
        """Process all DFT calculations and save graphs list to a .pt file."""
        self.graphs = []

        for root, dirs, files in os.walk(self.dft_data_path):
            poscar = os.path.join(root, 'POSCAR')
            outcar = os.path.join(root, 'OUTCAR')
            if not (os.path.exists(poscar) and os.path.exists(outcar)):
                continue

            lattice, positions, nB, nC, total = self.read_POSCAR(poscar)
            energy, forces = self.extract_OUTCAR(outcar)
            if energy is None or not forces:
                continue

            atom_types = ["B"] * nB + ["C"] * nC
            graph = self.build_graph(positions, atom_types, forces, energy, lattice)
            if graph is not None:
                self.graphs.append(graph)
                print(f"Created graph with {graph.edge_index.shape[1]} edges")

        # Save all graphs to a PyTorch file instead of Excel
        output_file = os.path.join(self.dft_data_path, 'graphs.pt') # .pt file saving for Bulk Export
        torch.save(self.graphs, output_file)
        print(f"Saved {len(self.graphs)} graphs to {output_file}")

        return self.graphs
    
    def process_folder(self, folder_path):
        """Process one DFT folder and return a single graph"""
        poscar = os.path.join(folder_path, 'POSCAR')
        outcar = os.path.join(folder_path, 'OUTCAR')

        if not (os.path.exists(poscar) and os.path.exists(outcar)):
            return None

        lattice, positions, nB, nC, total = self.read_POSCAR(poscar)
        energy, forces = self.extract_OUTCAR(outcar)

        if energy is None or not forces:
            return None

        atom_types = ["B"] * nB + ["C"] * nC
        graph = self.build_graph(positions, atom_types, forces, energy, lattice)
        return graph
