"""
Phase 4, step 1: build one network per target average degree, save as edgelist.
Thin CLI wrapper around common.graphs.build_graph (called by run.sh).
"""

import os, sys, argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.graphs import build_graph, write_edgelist

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--avg-degree", type=int, required=True)
    ap.add_argument("--type", required=True, choices=["ER", "BA"])
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    G = build_graph(a.type, a.n, a.avg_degree, seed=a.seed)
    path = os.path.join(a.output_dir, f"graph_N{a.n}_k{a.avg_degree}.edgelist")
    write_edgelist(G, path)
    print(f"  k={a.avg_degree}: N={G.number_of_nodes()}, edges={G.number_of_edges()}")
