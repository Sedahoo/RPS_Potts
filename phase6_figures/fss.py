"""
Phase 6b: Finite-Size Scaling (FSS).

A true phase transition is razor-sharp only in an INFINITE system. On a finite
network the m_psi(eps) curve is ROUNDED near the transition -- and it gets
sharper as N grows. Here we sweep eps near the boundary for several N at fixed
<k>, and watch the curve steepen. This is the original FSS experiment.
"""

import os, sys, subprocess
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "..", "phase3_cpp", "mc_simulation")


def mc_curve(N, k, eps_vals, T=0.65, sweeps=1200, seed=1):
    G = nx.erdos_renyi_graph(N, k / (N - 1), seed=seed)
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G = nx.convert_node_labels_to_integers(G)
    edgelist = os.path.join(HERE, f"_fss_N{N}.edgelist")
    nx.write_edgelist(G, edgelist, data=False)
    burn = int(sweeps * 0.3)
    out = np.empty(len(eps_vals))
    for i, eps in enumerate(eps_vals):
        r = subprocess.run(
            [BIN, "--graph", edgelist, "--epsilon", str(eps), "--temp", str(T),
             "--sweeps", str(sweeps), "--burn-in", str(burn), "--seed", str(seed)],
            capture_output=True, text=True).stdout
        out[i] = float(r.split()[0])
    os.remove(edgelist)
    return out


def main():
    k = 10
    eps_vals = np.linspace(0.35, 0.75, 21)   # zoom on the transition band
    Ns = [200, 500, 1000, 2000]

    plt.figure(figsize=(8, 5.5))
    for N in Ns:
        m = mc_curve(N, k, eps_vals)
        plt.plot(eps_vals, m, "o-", ms=3, label=f"N = {N}")
        print(f"  N={N:>4} done")
    plt.axhline(0.5, color="gray", ls="--", lw=1)
    plt.xlabel("epsilon"); plt.ylabel(r"$m_\psi$")
    plt.title(f"Phase 6b: finite-size scaling (ER, <k>={k}, T=0.65)\n"
              "the transition sharpens as N grows")
    plt.ylim(-0.02, 1.05); plt.legend(title="system size")
    plt.tight_layout()
    plt.savefig("fss.png", dpi=130)
    print("Saved fss.png")


if __name__ == "__main__":
    main()
