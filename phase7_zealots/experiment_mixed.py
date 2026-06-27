"""
Phase 7c: two COMPETING zealot factions (Rock vs Paper).

We add equal fractions z of Rock-zealots AND Paper-zealots (total 2z) and grow z.
From phase 7 we know a single faction provokes its predator: Rock-zealots push
free nodes to Paper, Paper-zealots push free nodes to Scissors. So with both
present, where does the free population go -- do they cancel into disorder, or
does the "predator of the predators" (Scissors) take over?

We plot the global strategy fractions and m_psi vs z, in both phases.
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.graphs import build_graph, write_edgelist
from common import runner

N, K = 800, 10
SEEDS = list(range(1, 13))
HERE = os.path.dirname(__file__)


def main():
    runner.ensure_engine()
    edgelists = [write_edgelist(build_graph("ER", N, K, seed=s),
                                os.path.join(HERE, f"_mx_{s}.edgelist")) for s in SEEDS]
    z_vals = np.linspace(0.0, 0.10, 16)        # each faction gets z (total 2z)
    regimes = [("Ordering phase (eps=0.3)", 0.3), ("Cycling phase (eps=0.9)", 0.9)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (title, eps) in zip(axes, regimes):
        jobs = [dict(edgelist=el, eps=eps,
                     zealot_frac=float(z), zealot_strategy=0,     # Rock faction
                     zealot_frac_b=float(z), zealot_strategy_b=1, # Paper faction
                     seed=s)
                for z in z_vals for el, s in zip(edgelists, SEEDS)]
        res = runner.run_many(jobs)
        def avg(key):
            return np.array([r[key] for r in res]).reshape(len(z_vals), len(SEEDS)).mean(axis=1)
        r, p, s, m = avg("r"), avg("p"), avg("s"), avg("m_psi")

        ax.plot(z_vals, r, "o-", color="tab:red",    label=r"$\rho_{rock}$ (incl. zealots)")
        ax.plot(z_vals, p, "o-", color="tab:green",  label=r"$\rho_{paper}$ (incl. zealots)")
        ax.plot(z_vals, s, "o-", color="tab:blue",   label=r"$\rho_{scissors}$ (free only)")
        ax.plot(z_vals, m, "s-", color="black",      label=r"$m_\psi$")
        ax.plot(z_vals, z_vals, ":", color="tab:red",   lw=1)   # rock zealot floor
        ax.plot(z_vals, z_vals, ":", color="tab:green", lw=1)   # paper zealot floor (same)
        ax.set_title(title); ax.set_xlabel("each faction's fraction z (total 2z)")
        ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel("global fraction")
    fig.suptitle("Phase 7c: competing Rock + Paper zealots -> does Scissors win?",
                 fontweight="bold")
    fig.tight_layout()
    for el in edgelists:
        os.remove(el)
    fig.savefig("zealots_mixed.png", dpi=130)
    print("Saved zealots_mixed.png")


if __name__ == "__main__":
    main()
