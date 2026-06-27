"""
Phase 3 validation: prove the C++ port matches the Python MC, and measure how
much faster it is.

We build ONE graph, write it as an edgelist, then run both engines on it for a
few epsilon values. The RNGs differ, so m_psi won't match to the digit -- but
it must agree within stochastic noise (and nail the ordered/disordered call).
The timing print is the payoff of the whole phase.
"""

import os, sys, time, subprocess
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase2_mc"))
from mc import build_graph, run_mc, order_parameter

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "mc_simulation")
EDGELIST = os.path.join(HERE, "g.edgelist")

N, K, T, SWEEPS = 500, 10, 0.65, 1500


def write_edgelist():
    p = K / (N - 1)
    G = nx.erdos_renyi_graph(N, p, seed=42)
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G = nx.convert_node_labels_to_integers(G)
    nx.write_edgelist(G, EDGELIST, data=False)
    return [list(G.neighbors(i)) for i in range(G.number_of_nodes())]


def main():
    neighbors = write_edgelist()
    burn = int(SWEEPS * 0.3)
    print(f"{'eps':>5} | {'Python m_psi':>12} {'C++ m_psi':>10} | {'verdict':>11}")
    print("-" * 50)

    t_py = t_cpp = 0.0
    for eps in (0.2, 0.5, 0.7, 0.9):
        t0 = time.time()
        py = order_parameter(run_mc(neighbors, eps, T=T, sweeps=SWEEPS, seed=7))
        t_py += time.time() - t0

        t0 = time.time()
        out = subprocess.run(
            [BIN, "--graph", EDGELIST, "--epsilon", str(eps), "--temp", str(T),
             "--sweeps", str(SWEEPS), "--burn-in", str(burn), "--seed", "7"],
            capture_output=True, text=True).stdout
        t_cpp += time.time() - t0
        cpp = float(out.split()[0])

        same = ("agree" if (py > 0.5) == (cpp > 0.5) else "DIFFER")
        print(f"{eps:>5} | {py:>12.3f} {cpp:>10.3f} | {same:>11}")

    print("-" * 50)
    print(f"Python total: {t_py:6.2f}s")
    print(f"C++ total:    {t_cpp:6.2f}s")
    print(f"Speedup:      {t_py/max(t_cpp,1e-9):6.1f}x")


if __name__ == "__main__":
    main()
