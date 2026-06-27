"""
Phase 4, step 1: build one network per target average degree and save it as an
edgelist the C++ engine can read.

Same logic as the original repo's generate_graphs.py: ER uses p = <k>/(N-1);
BA adds m = <k>/2 edges per node. We keep only the giant connected component
so no agent is left with zero neighbours.
"""

import argparse, os
import networkx as nx


def make_graph(n, avg_degree, graph_type, seed):
    if graph_type == "ER":
        G = nx.erdos_renyi_graph(n, avg_degree / (n - 1), seed=seed)
    elif graph_type == "BA":
        G = nx.barabasi_albert_graph(n, max(1, avg_degree // 2), seed=seed)
    else:
        raise ValueError("graph_type must be ER or BA")
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return nx.convert_node_labels_to_integers(G)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--avg-degree", type=int, required=True)
    ap.add_argument("--type", required=True, choices=["ER", "BA"])
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    G = make_graph(a.n, a.avg_degree, a.type, a.seed)
    path = os.path.join(a.output_dir, f"graph_N{a.n}_k{a.avg_degree}.edgelist")
    nx.write_edgelist(G, path, data=False)
    print(f"  k={a.avg_degree}: N={G.number_of_nodes()}, edges={G.number_of_edges()}")
