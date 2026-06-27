"""Network construction and degree bookkeeping (shared by every phase)."""

import os
from collections import Counter
import networkx as nx


def build_graph(graph_type, n, avg_degree, seed=1):
    """Build an ER or BA graph, keep its giant component, relabel 0..N-1.

    ER:  edge probability p = <k>/(N-1).
    BA:  m = <k>/2 edges added per node  (so <k> ~ 2m).
    """
    if graph_type == "ER":
        G = nx.erdos_renyi_graph(n, avg_degree / (n - 1), seed=seed)
    elif graph_type == "BA":
        G = nx.barabasi_albert_graph(n, max(1, avg_degree // 2), seed=seed)
    else:
        raise ValueError("graph_type must be 'ER' or 'BA'")
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return nx.convert_node_labels_to_integers(G)


def remove_edges(G, frac, seed=0):
    """Quench a fraction of edges (disabled connections). Keeps the giant
    component so the simulation stays well-defined."""
    import random
    if frac <= 0:
        return G.copy()
    rng = random.Random(seed)
    H = G.copy()
    edges = list(H.edges())
    rng.shuffle(edges)
    H.remove_edges_from(edges[:int(frac * len(edges))])
    H = H.subgraph(max(nx.connected_components(H), key=len)).copy()
    return nx.convert_node_labels_to_integers(H)


def remove_nodes(G, frac, seed=0):
    """Quench a fraction of nodes (vacancies). Keeps the giant component."""
    import random
    if frac <= 0:
        return G.copy()
    rng = random.Random(seed)
    H = G.copy()
    nodes = list(H.nodes())
    rng.shuffle(nodes)
    H.remove_nodes_from(nodes[:int(frac * len(nodes))])
    H = H.subgraph(max(nx.connected_components(H), key=len)).copy()
    return nx.convert_node_labels_to_integers(H)


def write_edgelist(G, path):
    """Write an undirected edgelist the C++ engine can read; return the path."""
    nx.write_edgelist(G, path, data=False)
    return path


def neighbor_lists(G):
    """Adjacency as plain Python lists (fast to iterate in the pure-Python MC)."""
    return [list(G.neighbors(i)) for i in range(G.number_of_nodes())]


def degree_dist(G):
    """Return (pk, mean_degree): pk = {degree -> fraction of nodes}."""
    degs = [d for _, d in G.degree()]
    nnow = G.number_of_nodes()
    pk = {k: c / nnow for k, c in Counter(degs).items()}
    mean_deg = sum(degs) / nnow
    return pk, mean_deg
