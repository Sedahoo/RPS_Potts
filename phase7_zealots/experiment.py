"""
Phase 7 experiment: can a few zealots take over the network?

We fix an ER graph and add a growing fraction z of zealots, all locked to Rock.
We do this in BOTH phases:
  * ordering phase (low eps): the network already wants consensus -- do zealots
    just steer which strategy wins?
  * cycling phase (high eps): normally no strategy dominates -- can a small z
    BREAK the cyclic symmetry and pin the network to Rock?

For each z we record, averaged over a few seeds:
  * conversion = fraction of the FREE (non-zealot) nodes that play Rock.
  * m_psi      = global order parameter (did the network order at all?).

The dashed grey line is the "do nothing" baseline conversion = 1/3 (what you'd
get if free nodes ignored the zealots and split evenly).
"""

import os, sys, subprocess
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "mc_zealots")
ZS = 0  # zealots play Rock


def make_graph(n, k, seed):
    G = nx.erdos_renyi_graph(n, k / (n - 1), seed=seed)
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G = nx.convert_node_labels_to_integers(G)
    path = os.path.join(HERE, f"_zg_{seed}.edgelist")
    nx.write_edgelist(G, path, data=False)
    return path


def run(edgelist, eps, z, seed, T=0.65, sweeps=1500):
    out = subprocess.run(
        [BIN, "--graph", edgelist, "--epsilon", str(eps), "--temp", str(T),
         "--sweeps", str(sweeps), "--burn-in", str(int(sweeps*0.3)),
         "--zealot-frac", str(z), "--zealot-strategy", str(ZS), "--seed", str(seed)],
        capture_output=True, text=True).stdout.split()
    return float(out[0]), float(out[4])   # m_psi, conversion


def main():
    N, k = 800, 10
    z_vals = np.linspace(0.0, 0.20, 17)
    regimes = [("Ordering phase (eps=0.3)", 0.3), ("Cycling phase (eps=0.9)", 0.9)]
    seeds = [1, 2, 3]
    edgelists = [make_graph(N, k, s) for s in seeds]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (title, eps) in zip(axes, regimes):
        conv = np.zeros(len(z_vals)); mpsi = np.zeros(len(z_vals))
        for i, z in enumerate(z_vals):
            cs, ms = [], []
            for el, s in zip(edgelists, seeds):
                m, c = run(el, eps, z, s)
                ms.append(m); cs.append(c)
            conv[i] = np.mean(cs); mpsi[i] = np.mean(ms)
        ax.plot(z_vals, conv, "o-", color="tab:red",
                label="conversion (free nodes playing Rock)")
        ax.plot(z_vals, mpsi, "s-", color="tab:purple", label=r"$m_\psi$ (global order)")
        ax.axhline(1/3, color="gray", ls="--", lw=1, label="no-influence baseline (1/3)")
        ax.plot([0, 0.2], [0, 0.2], color="gray", ls=":", lw=1, label="conversion = z")
        ax.set_title(title); ax.set_xlabel("zealot fraction z")
        ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=8, loc="center right")
    axes[0].set_ylabel("fraction")
    fig.suptitle("Phase 7: can Rock-zealots take over the network?",
                 fontweight="bold")
    fig.tight_layout()
    for el in edgelists:
        os.remove(el)
    fig.savefig("zealots.png", dpi=130)
    print("Saved zealots.png")


if __name__ == "__main__":
    main()
