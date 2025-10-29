#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import torch

def load_first_data(graphs_pt):
    obj = torch.load(graphs_pt, map_location="cpu", weights_only=False)
    if isinstance(obj, list):
        if len(obj) == 0:
            raise ValueError("empty graphs.pt list")
        return obj[0]
    return obj

def count_BC(z_tensor):
    n_B = int((z_tensor == 5).sum().item())
    n_C = int((z_tensor == 6).sum().item())
    return n_B, n_C

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("root_dir", help="Root folder to scan (recursively)")
    p.add_argument("--output", required=True, help="Path to write CSV")
    p.add_argument("--is-eq", type=int, default=0, choices=[0, 1],
                   help="Set 1 for equilibrium data (default 0)")
    p.add_argument("--graph-filename", default="graphs.pt",
                   help="Filename to look for (default: graphs.pt)")
    # NEW: include/exclude by first-level directory name under root_dir
    p.add_argument("--include-first", nargs="*", default=None,
                   help="Only include these first-level dirs (e.g. 15 30 60 120)")
    p.add_argument("--exclude-first", nargs="*", default=None,
                   help="Exclude these first-level dirs (e.g. Test_Data)")
    return p.parse_args()

def main():
    args = parse_args()
    root_dir = os.path.abspath(args.root_dir)
    rows = []

    for dirpath, _, filenames in os.walk(root_dir):
        if args.graph_filename not in filenames:
            continue

        full_path = os.path.join(dirpath, args.graph_filename)
        rel = os.path.relpath(full_path, root_dir).replace("\\", "/")

        # First-level folder under root (e.g., '120', 'Test_Data', etc.)
        first = rel.split("/", 1)[0]

        # Apply include/exclude filters if provided
        if args.include_first is not None and first not in set(args.include_first):
            continue
        if args.exclude_first is not None and first in set(args.exclude_first):
            continue

        try:
            data = load_first_data(full_path)
            if not hasattr(data, "z") or not hasattr(data, "pos"):
                raise ValueError("Data missing z or pos")
            n_atoms = int(data.pos.size(0))
            n_B, n_C = count_BC(data.z)
            comp = f"B{n_B}C{n_C}"
            tag = os.path.basename(os.path.dirname(full_path))

            rows.append(dict(
                graph_path=rel,                        # includes size folder when scanning parent
                is_eq=int(args.is_eq),
                n_B=n_B,
                n_C=n_C,
                n_atoms=n_atoms,
                composition=comp,
                tag=tag,
            ))
        except Exception as e:
            print(f"[WARN] skip {full_path}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"[OK] wrote {len(df)} rows -> {args.output}")

if __name__ == "__main__":
    main()
