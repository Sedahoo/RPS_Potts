"""
Phase 7b: do zealots on BA HUBS punch above their weight?

Same fraction z of Rock-zealots placed on RANDOM nodes vs the HIGHEST-DEGREE
nodes of a BA graph. Hubs touch many free nodes, so they should break the cyclic
symmetry at much smaller z. Finding: in the cycling phase hub placement amplifies
the effect ~8x (m_psi), but the induced order is still Paper (the predator), not
the zealots' Rock.
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.graphs import build_graph, write_edgelist
from common import runner

N, K = 800, 10


def main():
    runner.ensure_engine()
    seeds = [1, 2, 3]
    edgelists = [write_edgelist(build_graph("BA", N, K, seed=s),
                                os.path.join(os.path.dirname(__file__), f"_hub_{s}.edgelist"))
                 for s in seeds]
    z_vals = np.linspace(0.0, 0.10, 16)
    regimes = [("Ordering phase (eps=0.3)", 0.3), ("Cycling phase (eps=0.9)", 0.9)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (title, eps) in zip(axes, regimes):
        for target, col in [("random", "tab:blue"), ("hub", "tab:orange")]:
            mp = np.zeros(len(z_vals)); cv = np.zeros(len(z_vals))
            for i, z in enumerate(z_vals):
                res = [runner.run_engine(el, eps, zealot_frac=z, zealot_strategy=0,
                                         zealot_target=target, seed=s)
                       for el, s in zip(edgelists, seeds)]
                mp[i] = np.mean([r["m_psi"] for r in res])
                cv[i] = np.mean([r["conversion"] for r in res])
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
