import os
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


def read_POSCAR(file_path):
    """
    Reads a POSCAR file and returns:
      - lattice_vectors: (3,3) array
      - atom_positions:  (N,3) Cartesian positions
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Line 2: scaling factor
    scale = float(lines[1].strip())

    # Lines 3-5: lattice vectors
    lattice_vectors = np.array([list(map(float, lines[i].split())) for i in range(2, 5)]) * scale

    # Line 7: atom counts (n_boron n_carbon)
    atom_counts = lines[6].split()
    n_boron  = int(atom_counts[0])
    n_carbon = int(atom_counts[1])
    total_atoms = n_boron + n_carbon

    # Read atomic positions (lines 9 to 9+total_atoms-1)
    pos_start = 8
    positions = []
    for i in range(pos_start, pos_start + total_atoms):
        parts = lines[i].split()
        positions.append(list(map(float, parts[:3])))
    atom_positions = np.array(positions)

    # Check for "Direct" coordinates
    coord_type = lines[pos_start - 1].strip().lower()
    if "direct" in coord_type:
        # Convert from fractional to Cartesian
        atom_positions = atom_positions.dot(lattice_vectors)

    return lattice_vectors, atom_positions

def compute_distances(lattice, positions):
    inv_lat = np.linalg.inv(lattice)
    frac = positions.dot(inv_lat.T)
    dists = []
    N = len(positions)
    for i in range(N):
        for j in range(i+1, N):
            delta = frac[j] - frac[i]
            delta -= np.round(delta)
            vec = lattice.dot(delta)
            dists.append(np.linalg.norm(vec))
    return dists

def main(root_dir, cutoff, max_search=10.0):
    # 1) Gather all distances
    all_dists = []
    for subdir, _, files in os.walk(root_dir):
        if 'POSCAR' in files:
            lattice, pos = read_POSCAR(os.path.join(subdir, 'POSCAR'))
            all_dists.extend(compute_distances(lattice, pos))
    all_dists = np.array(all_dists)

    # 2) Histogram & bin centers
    hist, edges = np.histogram(all_dists, bins=500, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # 3) Trim to <= max_search Å
    mask   = centers <= max_search
    hist_t = hist[mask]
    ctr_t  = centers[mask]

    # 4) Find last peak
    peaks     = (hist_t[1:-1] > hist_t[:-2]) & (hist_t[1:-1] > hist_t[2:])
    peak_idxs = np.where(peaks)[0] + 1
    last_peak = peak_idxs.max()

    # 5) Pick first drop below 10% of peak height
    threshold = hist_t.max() * 0.10
    tail = hist_t[last_peak+1:]
    idxs = np.where(tail < threshold)[0]
    if idxs.size > 0:
        valley_idx = last_peak + 1 + idxs[0]
    else:
        # fallback to local-minimum logic
        mins     = (hist_t[1:-1] < hist_t[:-2]) & (hist_t[1:-1] < hist_t[2:])
        min_idxs = np.where(mins)[0] + 1
        val_after = min_idxs[min_idxs > last_peak]
        valley_idx = val_after.min() if val_after.size > 0 else last_peak

    suggested_rc = ctr_t[valley_idx]
    print(f"Suggested cutoff (10% threshold) ≃ {suggested_rc:.2f} Å")

    # 6) Plot
    plt.figure(figsize=(6,4))
    plt.hist(all_dists, bins=500, density=True, alpha=0.8)
    plt.axvline(cutoff, linestyle='--', color='red',   label=f'current = {cutoff:.2f} Å')
    plt.axvline(suggested_rc, linestyle='--', color='green', label=f'suggested = {suggested_rc:.2f} Å')
    plt.xlabel("Distance (Å)")
    plt.ylabel("Normalized count")
    plt.title("Pairwise distance distribution across dataset")
    plt.legend(fontsize='small')
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot RDF & suggest cutoff")
    parser.add_argument("--root",   required=True, help="Root dir of POSCAR folders")
    parser.add_argument("--cutoff", type=float, default=6.3, help="Current cutoff for comparison")
    parser.add_argument("--max",    type=float, default=10.0, help="Max distance to consider (Å)")
    args = parser.parse_args()
    main(args.root, args.cutoff, max_search=args.max)