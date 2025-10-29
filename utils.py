# --- utils.py ------------------------------------------------------------
import torch
from torch_geometric.nn import radius_graph

def radius_graph_pbc(pos_cart, lattice, cutoff, max_num_neigh=128):
    """
    Build a radius graph with ±1 cell images.
    Returns
        edge_index  : [2, E]  indices in 0..N-1
        edge_shift  : [E, 3]  integer multiples of lattice vectors
    """
    device = pos_cart.device
    N      = pos_cart.size(0)

    # 1) generate 27 periodic images
    shifts = torch.tensor([[i, j, k]
                           for i in (-1, 0, 1)
                           for j in (-1, 0, 1)
                           for k in (-1, 0, 1)],
                          dtype=torch.long,
                          device=device)             # [27,3]

    # 2) replicate positions across images
    pos_images = pos_cart[None, :, :] + (shifts.to(pos_cart.dtype)[:, None, :] @ lattice)  # [27, N, 3]
    pos_images = pos_images.view(-1, 3)                                 # [27*N, 3]

    atom_id  = torch.arange(N, device=device).repeat(27)                # [27*N]
    shift_id = torch.arange(27, device=device).repeat_interleave(N)     # [27*N]

    # 3) build initial radius graph (Cartesian)
    edge_img = radius_graph(pos_images,
                            r=cutoff,
                            max_num_neighbors=max_num_neigh,
                            loop=False)
    src_img, dst_img = edge_img

    # map back to original atoms and shifts
    src_atom   = atom_id[src_img]
    dst_atom   = atom_id[dst_img]
    edge_shift = shifts[shift_id[dst_img]]                              # [E,3]

    # drop duplicate self‐edges from the zero‐shift image
    mask = (edge_shift != 0).any(dim=-1) | (dst_img != src_img)
    edge_index = torch.stack([src_atom[mask], dst_atom[mask]], dim=0)
    edge_shift = edge_shift[mask]                                       # [E,3]

    # --- NEW: filter by true PBC distance ---
    src, dst = edge_index
    coords_src = pos_cart[src]                                          # [E,3]
    coords_dst = pos_cart[dst] + (edge_shift.to(pos_cart.dtype) @ lattice)                 # [E,3]
    dists = (coords_dst - coords_src).norm(dim=1)                       # [E]
    keep = dists <= cutoff + 1e-6
    edge_index = edge_index[:, keep]
    edge_shift = edge_shift[keep]
    edge_shift = edge_shift.to(pos_cart.dtype)
    # -----------------------------------------

    return edge_index, edge_shift
