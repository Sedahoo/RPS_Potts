"""
Phase 7b: do zealots on BA HUBS punch above their weight?

BA graphs have a few very-high-degree hubs; ER graphs don't. We place the same
fraction z of Rock-zealots either on RANDOM nodes or on the HIGHEST-DEGREE nodes
of a BA graph, and compare their effect in both phases.

Hypothesis: a hub touches a huge number of free nodes, so hub-zealots should
exert outsized influence -- breaking the cyclic symmetry (and triggering the
"flip to Paper" effect from phase 7) at a much smaller z than random placement.

We track, vs z (averaged over a few BA graphs):
  * m_psi      -- global order (did the cycle break?)
  * conversion -- fraction of FREE nodes playing Rock (the zealots' strategy)
"""

import os, sys, subprocess
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "mc_zealots")


def make_ba(n, k, seed):
    G = nx.barabasi_albert_graph(n, max(1, k // 2), seed=seed)
    G = nx.convert_node_labels_to_integers(G)
    path = os.path.join(HERE, f"_hub_{seed}.edgelist")
    nx.write_edgelist(G, path, data=False)
    return path


def run(edgelist, eps, z, target, seed, T=0.65, sweeps=1500):
    out = subprocess.run(
        [BIN, "--graph", edgelist, "--epsilon", str(eps), "--temp", str(T),
         "--sweeps", str(sweeps), "--burn-in", str(int(sweeps*0.3)),
         "--zealot-frac", str(z), "--zealot-strategy", "0",
         "--zealot-target", target, "--seed", str(seed)],
        capture_output=True, text=True).stdout.split()
    return float(out[0]), float(out[4])   # m_psi, conversion


def main():
    N, k = 800, 10
    z_vals = np.linspace(0.0, 0.10, 16)
    seeds = [1, 2, 3]
    edgelists = [make_ba(N, k, s) for s in seeds]
    regimes = [("Ordering phase (eps=0.3)", 0.3), ("Cycling phase (eps=0.9)", 0.9)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (title, eps) in zip(axes, regimes):
        for target, col in [("random", "tab:blue"), ("hub", "tab:orange")]:
            mp = np.zeros(len(z_vals)); cv = np.zeros(len(z_vals))
            for i, z in enumerate(z_vals):
                res = [run(el, eps, z, target, s) for el, s in zip(edgelists, seeds)]
                mp[i] = np.mean([r[0] for r in res])
                cv[i] = np.mean([r[1] for r in res])
            ax.plot(z_vals, mp, "s-", color=col, label=f"{target}: $m_\\psi$")
            ax.plot(z_vals, cv, "o--", color=col, alpha=0.6, label=f"{target}: conversion")
        ax.axhline(1/3, color="gray", ls=":", lw=1)
        ax.set_title(title); ax.set_xlabel("zealot fraction z")
        ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=8, loc="center right")
    axes[0].set_ylabel("fraction")
    fig.suptitle("Phase 7b: hub-placed vs random Rock-zealots on a BA network",
                 fontweight="bold")
    fig.tight_layout()
    for el in edgelists:
        os.remove(el)
    fig.savefig("zealots_hubs.png", dpi=130)
    print("Saved zealots_hubs.png")


if __name__ == "__main__":
    main()
