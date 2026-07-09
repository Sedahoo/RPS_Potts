"""
Sec. 6.1 across the parameter space: single Rock-zealot faction at multiple
T, <k> and N (one axis varied at a time around the base point ER, N=800,
<k>=10, T=0.65 of experiment.py).

Hypotheses (embedded in the report): the backfire (free nodes adopt Paper,
Rock's predator) is generic -- it should survive every cell. T weakens the
ferromagnetic pinning, so hotter systems order less sharply on the beater and
the large-z re-pinning on Rock needs more zealots. <k> strengthens the local
majority pressure (effective noise ~T/k), so denser graphs show a cleaner
backfire. N is a null axis: conversion and m_psi are intensive, so curves
should coincide within noise, only smoother at larger N.
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.graphs import build_graph, write_edgelist
from common.io import save_table
from common import runner

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = dict(T=0.65, k=10, N=800)
AXES = [("T", [0.40, 0.65, 1.00]), ("k", [6, 10, 20]), ("N", [400, 800, 1600])]
ZV = np.linspace(0.0, 0.20, 9)
SEEDS = list(range(1, 9))
REGIMES = [(0.3, "ordering"), (0.9, "cycling")]


def cells():
    """The 7 distinct (T,k,N) cells: base + one-at-a-time variations."""
    seen, out = set(), []
    for name, vals in AXES:
        for v in vals:
            c = dict(BASE); c[name] = v
            key = (c["T"], c["k"], c["N"])
            if key not in seen:
                seen.add(key); out.append(c)
    return out


def main():
    runner.ensure_engine()
    cs = cells()
    els = {}
    for c in cs:
        for s in SEEDS:
            key = (c["k"], c["N"], s)
            if key not in els:
                els[key] = write_edgelist(
                    build_graph("ER", c["N"], c["k"], seed=s),
                    os.path.join(HERE, f"_gg_{c['k']}_{c['N']}_{s}.edgelist"))
    points = [(c, eps, z, s) for c in cs for eps, _ in REGIMES for z in ZV
              for s in SEEDS]
    res = runner.run_many([dict(edgelist=els[(c["k"], c["N"], s)], eps=eps,
                                temp=c["T"], zealot_frac=float(z),
                                zealot_strategy=0, seed=s)
                           for c, eps, z, s in points])
    for el in els.values():
        os.remove(el)

    long = {k_: [] for k_ in ("T", "k", "N", "epsilon", "z", "conversion",
                              "conversion_sd", "m_psi", "m_psi_sd", "rho_paper")}
    nz, ns = len(ZV), len(SEEDS)
    block = 2 * nz * ns                      # results per cell
    arr = {c_i: res[c_i * block:(c_i + 1) * block] for c_i in range(len(cs))}
    for c_i, c in enumerate(cs):
        for r_i, (eps, _) in enumerate(REGIMES):
            for z_i, z in enumerate(ZV):
                chunk = arr[c_i][r_i * nz * ns + z_i * ns:(r_i * nz * ns
                                                           + (z_i + 1) * ns)]
                conv = np.array([r["conversion"] for r in chunk])
                mp = np.array([r["m_psi"] for r in chunk])
                pap = np.array([r["p"] for r in chunk])
                long["T"].append(c["T"]); long["k"].append(c["k"])
                long["N"].append(c["N"]); long["epsilon"].append(eps)
                long["z"].append(z)
                long["conversion"].append(conv.mean())
                long["conversion_sd"].append(conv.std())
                long["m_psi"].append(mp.mean()); long["m_psi_sd"].append(mp.std())
                long["rho_paper"].append(pap.mean())
    save_table(os.path.join(HERE, "zealots_grid.csv"), long)

    lg = {k_: np.array(v) for k_, v in long.items()}
    fig, axes = plt.subplots(len(AXES), 2, figsize=(12, 11), sharex=True)
    for row, (name, vals) in enumerate(AXES):
        cmap = plt.cm.viridis(np.linspace(0, 0.85, len(vals)))
        for col, (eps, rname) in enumerate(REGIMES):
            ax = axes[row, col]
            for v_i, v in enumerate(vals):
                c = dict(BASE); c[name] = v
                sel = ((lg["T"] == c["T"]) & (lg["k"] == c["k"])
                       & (lg["N"] == c["N"]) & (lg["epsilon"] == eps))
                y = lg["conversion"] if col == 0 else lg["m_psi"]
                ysd = lg["conversion_sd"] if col == 0 else lg["m_psi_sd"]
                ax.errorbar(lg["z"][sel], y[sel], yerr=ysd[sel], fmt="o-", ms=3,
                            capsize=2, color=cmap[v_i], label=f"{name}={v:g}")
                if col == 0:
                    ax.plot(lg["z"][sel], lg["rho_paper"][sel], "--", lw=1,
                            color=cmap[v_i], alpha=0.7)
            ax.axhline(1 / 3, color="gray", ls=":", lw=0.8)
            ax.legend(fontsize=8)
            if row == 0:
                ax.set_title(f"{rname} phase ($\\varepsilon{{=}}{eps}$): "
                             + ("conversion (dashed: $\\rho_{Paper}$)"
                                if col == 0 else "$m_\\psi$"))
        axes[row, 0].set_ylabel(f"vary {name}")
    for ax in axes[-1]:
        ax.set_xlabel("zealot fraction z")
    fig.suptitle("Sec. 6.1 across $T$, $\\langle k\\rangle$, $N$: Rock-zealots, "
                 "one axis varied at a time around (ER, $N{=}800$, "
                 "$\\langle k\\rangle{=}10$, $T{=}0.65$)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "zealots_grid.png"), dpi=130)
    print("Saved zealots_grid.png")


if __name__ == "__main__":
    main()
