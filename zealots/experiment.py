"""
Phase 7 experiment: can a few Rock-zealots take over the network?

We grow the fraction z of zealots (locked to Rock) on a fixed ER graph, in both
the ordering phase (eps=0.3) and the cycling phase (eps=0.9), and record:
  * conversion = fraction of FREE nodes that play Rock
  * m_psi      = global order parameter
Finding: in the ordering phase a few Rock-zealots flip the free network to PAPER
(the strategy that beats Rock), not Rock; in the cycling phase they induce only
weak order and can't pin their own strategy.
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.graphs import build_graph, write_edgelist
from common import runner

N, K, ZS = 800, 10, 0      # zealots play Rock (strategy 0)
SEEDS = list(range(1, 13))  # 12 ER graphs to average over


def main():
    runner.ensure_engine()
    edgelists = [write_edgelist(build_graph("ER", N, K, seed=s),
                                os.path.join(os.path.dirname(__file__), f"_zg_{s}.edgelist"))
                 for s in SEEDS]
    z_vals = np.linspace(0.0, 0.20, 17)
    regimes = [("Ordering phase (eps=0.3)", 0.3), ("Cycling phase (eps=0.9)", 0.9)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (title, eps) in zip(axes, regimes):
        jobs = [dict(edgelist=el, eps=eps, zealot_frac=float(z), zealot_strategy=ZS, seed=s)
                for z in z_vals for el, s in zip(edgelists, SEEDS)]
        res = runner.run_many(jobs)
        conv = np.array([r["conversion"] for r in res]).reshape(len(z_vals), len(SEEDS)).mean(axis=1)
        mpsi = np.array([r["m_psi"] for r in res]).reshape(len(z_vals), len(SEEDS)).mean(axis=1)
        ax.plot(z_vals, conv, "o-", color="tab:red",
                label="conversion (free nodes playing Rock)")
        ax.plot(z_vals, mpsi, "s-", color="tab:purple", label=r"$m_\psi$ (global order)")
        ax.axhline(1/3, color="gray", ls="--", lw=1, label="no-influence baseline (1/3)")
        ax.plot([0, 0.2], [0, 0.2], color="gray", ls=":", lw=1, label="conversion = z")
        ax.set_title(title); ax.set_xlabel("zealot fraction z")
        ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=8, loc="center right")
    axes[0].set_ylabel("fraction")
    fig.suptitle("Phase 7: can Rock-zealots take over the network?", fontweight="bold")
    fig.tight_layout()
    for el in edgelists:
        os.remove(el)
    fig.savefig("zealots.png", dpi=130)
    print("Saved zealots.png")


if __name__ == "__main__":
    main()
