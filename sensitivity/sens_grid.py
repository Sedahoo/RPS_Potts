"""
S5 (report Sec. 4): eps-grid resolution and the eps_c estimator.

One fine sweep (step 0.0125, seed-averaged) is subsampled to coarser grids so
every resolution sees identical data and only the estimator's grid changes.
Hypothesis (sensitivity/HYPOTHESES.md): the interpolated estimator keeps the
eps_c shift well below half a step even at step 0.10; the naive
first-point-below estimator is biased by up to a full step.
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.graphs import build_graph, write_edgelist
from common.observables import eps_crossing
from common.io import save_table
from common import runner

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = np.linspace(0, 1, 81)              # step 0.0125
STRIDES = [1, 2, 4, 8]                   # -> steps 0.0125, 0.025, 0.05, 0.10
SEEDS = [1, 2, 3, 4]
N, K = 500, 20


def naive(e, m, thr=0.5):
    """The old convention: first grid point with m < thr."""
    below = np.where(np.asarray(m) < thr)[0]
    return float(e[below[0]]) if len(below) else float(e[-1])


def main():
    runner.ensure_engine()
    els = {s: write_edgelist(build_graph("ER", N, K, seed=s),
                             os.path.join(HERE, f"_gg_{s}.edgelist"))
           for s in SEEDS}
    res = runner.run_many([dict(edgelist=els[s], eps=float(e), seed=s)
                           for e in EPS for s in SEEDS])
    m = np.array([r["m_psi"] for r in res]).reshape(len(EPS), len(SEEDS)).mean(axis=1)
    for el in els.values():
        os.remove(el)

    steps, interp, naiv = [], [], []
    for st in STRIDES:
        e_sub, m_sub = EPS[::st], m[::st]
        steps.append(EPS[st] - EPS[0])
        interp.append(eps_crossing(e_sub, m_sub))
        naiv.append(naive(e_sub, m_sub))
    save_table(os.path.join(HERE, "sens_grid.csv"),
               {"step": steps, "eps_c_interp": interp, "eps_c_naive": naiv})
    save_table(os.path.join(HERE, "sens_grid_curves.csv"), {"epsilon": EPS, "m": m})

    ref = interp[0]
    for st, i, nv in zip(steps, interp, naiv):
        print(f"step={st:.4f}: interp={i:.4f} (shift {i-ref:+.4f})  "
              f"naive={nv:.4f} (shift {nv-ref:+.4f})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].plot(EPS, m, "-", color="tab:blue", lw=1.2, label="fine sweep (step 0.0125)")
    axes[0].axhline(0.5, color="gray", ls="--", lw=0.8)
    axes[0].set_xlabel(r"$\varepsilon$"); axes[0].set_ylabel(r"$m_\psi$")
    axes[0].set_title("seed-averaged fine sweep"); axes[0].legend()
    axes[1].plot(steps, np.abs(np.array(interp) - ref), "o-", color="tab:green",
                 label="interpolated (project estimator)")
    axes[1].plot(steps, np.abs(np.array(naiv) - ref), "s-", color="tab:red",
                 label="naive first-point-below")
    axes[1].plot(steps, np.array(steps) / 2, "k:", lw=1, label="half a grid step")
    axes[1].set_xlabel(r"grid step $\Delta\varepsilon$")
    axes[1].set_ylabel(r"$|\varepsilon_c - \varepsilon_c^{fine}|$")
    axes[1].set_title("estimator error vs resolution (same data)")
    axes[1].legend(fontsize=9)
    fig.suptitle("S5: $\\varepsilon$-grid resolution is a nuisance parameter "
                 "under interpolation", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "sens_grid.png"), dpi=130)
    print("Saved sens_grid.png")


if __name__ == "__main__":
    main()
