"""
S6 (report Sec. 6): zealot strategy label — an exact symmetry as a null test.

The zealot experiments lock zealots to Rock. The payoff matrix is exactly
symmetric under R->P->S->R, so relabeling the zealots must change NOTHING
statistically. Here the Sec. 6.1 protocol runs for all three labels, 32 graph
seeds (the seed also draws the placement), both phases; a paired permutation
test calibrates the label spread against its exact null. Hypothesis
(sensitivity/HYPOTHESES.md): label differences sit inside seed scatter at
every z; the backfire (free nodes adopt the strategy that BEATS the zealots)
appears for every label. Systematic label dependence = implementation bug.
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.graphs import build_graph, write_edgelist
from common.io import save_table
from common import runner

HERE = os.path.dirname(os.path.abspath(__file__))
N, K = 800, 10
STRATS = [0, 1, 2]                       # Rock, Paper, Scissors
NAMES = ["Rock", "Paper", "Scissors"]
KEYS = ["r", "p", "s"]
ZV = np.linspace(0.025, 0.20, 8)   # z=0 excluded: with no zealots the label is
                                   # undefined (all three would measure the SAME
                                   # run against different names)
SEEDS = list(range(1, 33))
REGIMES = [("ordering", 0.3), ("cycling", 0.9)]


def main():
    runner.ensure_engine()
    els = {s: write_edgelist(build_graph("ER", N, K, seed=s),
                             os.path.join(HERE, f"_zg_{s}.edgelist"))
           for s in SEEDS}
    points = [(eps, zs, z, s) for _, eps in REGIMES for zs in STRATS
              for z in ZV for s in SEEDS]
    res = runner.run_many([dict(edgelist=els[s], eps=eps, zealot_frac=float(z),
                                zealot_strategy=zs, seed=s)
                           for eps, zs, z, s in points])
    for el in els.values():
        os.remove(el)
    conv = np.array([r["conversion"] for r in res]).reshape(
        len(REGIMES), len(STRATS), len(ZV), len(SEEDS))
    mpsi = np.array([r["m_psi"] for r in res]).reshape(conv.shape)
    beat = np.array([res[i][KEYS[(p[1] + 1) % 3]]                 # rho of the label's beater
                     for i, p in enumerate(points)]).reshape(conv.shape)

    long = {"epsilon": [], "zealot_strategy": [], "z": [],
            "conversion": [], "conversion_sd": [],
            "m_psi": [], "m_psi_sd": [], "rho_beater": []}
    for i, (_, eps) in enumerate(REGIMES):
        for j, zs in enumerate(STRATS):
            for k, z in enumerate(ZV):
                long["epsilon"].append(eps); long["zealot_strategy"].append(zs)
                long["z"].append(z)
                long["conversion"].append(conv[i, j, k].mean())
                long["conversion_sd"].append(conv[i, j, k].std())
                long["m_psi"].append(mpsi[i, j, k].mean())
                long["m_psi_sd"].append(mpsi[i, j, k].std())
                long["rho_beater"].append(beat[i, j, k].mean())
    save_table(os.path.join(HERE, "sens_zealot_symmetry.csv"), long)

    # paired permutation test: under the symmetry null the three label values of
    # a given seed are exchangeable, so permuting labels WITHIN each seed gives
    # the exact null distribution of the label spread (max-min of label means).
    # Seed-level outcomes are bimodal at large z (a realisation either flips to
    # the zealot consensus or not), so this replaces a miscalibrated Gaussian test.
    prng = np.random.default_rng(0)
    n_perm = 2000
    ptab = {"epsilon": [], "z": [], "spread": [], "p_value": [], "coalesced": []}
    for i, (name, eps) in enumerate(REGIMES):
        pvals = []
        for k in range(len(ZV)):
            vals = conv[i, :, k, :]                        # (3 labels, seeds)
            obs = np.ptp(vals.mean(axis=1))
            null = np.empty(n_perm)
            for b in range(n_perm):
                idx = np.argsort(prng.random((vals.shape[1], 3)), axis=1)
                perm = vals[idx.T, np.arange(vals.shape[1])]
                null[b] = np.ptp(perm.mean(axis=1))
            p = float(np.mean(null >= obs - 1e-12))
            pvals.append(p)
            ptab["epsilon"].append(eps); ptab["z"].append(ZV[k])
            ptab["spread"].append(obs); ptab["p_value"].append(p)
            ptab["coalesced"].append(1.0 if obs == 0.0 else 0.0)
        pvals = np.array(pvals)
        n_sig = int(np.sum(pvals < 0.05))
        n_coalesced = int(np.sum(np.ptp(conv[i].mean(axis=2), axis=0) == 0.0))
        print(f"{name} (eps={eps}): permutation p-values per z = "
              f"{np.array2string(pvals, precision=2)}; {n_sig}/{len(ZV)} below 0.05; "
              f"{n_coalesced}/{len(ZV)} z-points with EXACTLY zero spread "
              f"(coalesced relabeled trajectories)")
    save_table(os.path.join(HERE, "sens_zealot_pvals.csv"), ptab)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    colors = ["tab:red", "tab:blue", "tab:green"]
    for ax, (i, (name, eps)) in zip(axes, enumerate(REGIMES)):
        for j, zs in enumerate(STRATS):
            ax.errorbar(ZV, conv[i, j].mean(axis=1), yerr=conv[i, j].std(axis=1),
                        fmt="o-", ms=4, capsize=2, color=colors[j],
                        label=f"{NAMES[j]}-zealots: conversion")
            ax.plot(ZV, beat[i, j].mean(axis=1), "--", lw=1, color=colors[j], alpha=0.7)
        ax.axhline(1 / 3, color="gray", ls=":", lw=1)
        ax.set_title(f"{name} phase ($\\varepsilon={eps}$)")
        ax.set_xlabel("zealot fraction z")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("fraction of free nodes")
    fig.suptitle("S6: the three zealot labels are statistically identical "
                 "(dashed: population of the label's beater)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "sens_zealot_symmetry.png"), dpi=130)
    print("Saved sens_zealot_symmetry.png")


if __name__ == "__main__":
    main()
