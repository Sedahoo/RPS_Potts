"""
Phase 7b: do zealots on BA HUBS punch above their weight?

Same fraction z of Rock-zealots placed on RANDOM nodes vs the HIGHEST-DEGREE
nodes of a BA graph, in both phases. Hubs touch many free nodes, so they should
break the cyclic symmetry at much smaller z.

Averaged over many BA graphs (seeds) so the ordering-phase consensus -- which is
multistable (the network locks onto SOME strategy) -- comes out as a smooth
probability rather than noise.

Findings: cycling phase -> hub placement amplifies the order induced ~8x vs
random; but the induced order is still Paper (the predator of Rock), not Rock.
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.graphs import build_graph, write_edgelist
from common import runner

N, K = 800, 10
SEEDS = list(range(1, 16))          # 15 BA graphs to average over
HERE = os.path.dirname(__file__)


def curve(edgelists, eps, z_vals, target):
    """Return (m_psi, conversion) averaged over seeds, for each z."""
    jobs = [dict(edgelist=el, eps=eps, zealot_frac=float(z), zealot_strategy=0,
                 zealot_target=target, seed=s)
            for z in z_vals for el, s in zip(edgelists, SEEDS)]
    res = runner.run_many(jobs)
    mp = np.array([r["m_psi"] for r in res]).reshape(len(z_vals), len(SEEDS))
    cv = np.array([r["conversion"] for r in res]).reshape(len(z_vals), len(SEEDS))
    return mp.mean(axis=1), cv.mean(axis=1)


def main():
    runner.ensure_engine()
    edgelists = [write_edgelist(build_graph("BA", N, K, seed=s),
                                os.path.join(HERE, f"_hub_{s}.edgelist")) for s in SEEDS]
    z_vals = np.linspace(0.0, 0.10, 16)
    regimes = [("Ordering phase (eps=0.3)", 0.3), ("Cycling phase (eps=0.9)", 0.9)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (title, eps) in zip(axes, regimes):
        for target, col in [("random", "tab:blue"), ("hub", "tab:orange")]:
            mp, cv = curve(edgelists, eps, z_vals, target)
            ax.plot(z_vals, mp, "s-", color=col, label=f"{target}: $m_\\psi$")
            ax.plot(z_vals, cv, "o--", color=col, alpha=0.6, label=f"{target}: conversion")
        ax.axhline(1/3, color="gray", ls=":", lw=1)
        ax.set_title(title); ax.set_xlabel("zealot fraction z")
        ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=8, loc="center right")
    axes[0].set_ylabel("fraction")
    fig.suptitle(f"Phase 7b: hub vs random Rock-zealots on BA "
                 f"(avg of {len(SEEDS)} graphs)", fontweight="bold")
    fig.tight_layout()
    for el in edgelists:
        os.remove(el)
    fig.savefig("zealots_hubs.png", dpi=130)
    print("Saved zealots_hubs.png")


if __name__ == "__main__":
    main()
