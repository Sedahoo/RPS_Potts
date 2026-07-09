"""
S4 (report Sec. 5): simulation length (sweeps) — does 1500 sweeps suffice?

Full eps sweeps at six run lengths (burn-in fixed at 30% of each), 4 graph
seeds; the 6000-sweep curve is the reference. Hypothesis
(sensitivity/HYPOTHESES.md): short runs bias the cycling phase UP (a short
window averages a fraction of a psi rotation) and may bias the ordered phase
down (unfinished ordering), so eps_c is overestimated at small sweeps and
converges from above by ~1500.
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
SWEEPS = [200, 500, 1000, 1500, 3000, 6000]
SEEDS = [1, 2, 3, 4]
EPS = np.linspace(0, 1, 21)
N, K = 500, 20


def main():
    runner.ensure_engine()
    els = {s: write_edgelist(build_graph("ER", N, K, seed=s),
                             os.path.join(HERE, f"_eg_{s}.edgelist"))
           for s in SEEDS}
    points = [(sw, e, s) for sw in SWEEPS for e in EPS for s in SEEDS]
    res = runner.run_many([dict(edgelist=els[s], eps=float(e), sweeps=sw, seed=s)
                           for sw, e, s in points])
    m = np.array([r["m_psi"] for r in res]).reshape(len(SWEEPS), len(EPS), len(SEEDS))
    for el in els.values():
        os.remove(el)

    mean = m.mean(axis=2)
    ref = mean[-1]
    summary = {"sweeps": SWEEPS,
               "eps_c": [eps_crossing(EPS, mean[i]) for i in range(len(SWEEPS))],
               "rmse_vs_ref": [float(np.sqrt(np.mean((mean[i] - ref) ** 2)))
                               for i in range(len(SWEEPS))],
               "m_cycle": [mean[i][EPS >= 0.8].mean() for i in range(len(SWEEPS))]}
    curves = {"epsilon": EPS}
    for i, sw in enumerate(SWEEPS):
        curves[f"m_sw{sw}"] = mean[i]
    save_table(os.path.join(HERE, "sens_equilibration.csv"), summary)
    save_table(os.path.join(HERE, "sens_equilibration_curves.csv"), curves)

    for i, sw in enumerate(SWEEPS):
        print(f"sweeps={sw:5d}: eps_c={summary['eps_c'][i]:.4f}  "
              f"RMSE vs 6000 = {summary['rmse_vs_ref'][i]:.4f}  "
              f"m_cycle={summary['m_cycle'][i]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    cmap = plt.cm.cividis(np.linspace(0, 0.9, len(SWEEPS)))
    for i, sw in enumerate(SWEEPS):
        axes[0].plot(EPS, mean[i], "o-", ms=3, color=cmap[i], label=f"{sw} sweeps")
    axes[0].axhline(0.5, color="gray", ls="--", lw=0.8)
    axes[0].set_xlabel(r"$\varepsilon$"); axes[0].set_ylabel(r"$m_\psi$")
    axes[0].set_title("short windows lift the cycling tail"); axes[0].legend(fontsize=8)
    ax2 = axes[1]
    ax2.semilogx(SWEEPS, summary["eps_c"], "o-", color="tab:blue",
                 label=r"$\varepsilon_c$")
    ax2.axvline(1500, color="gray", ls=":", lw=1)
    ax2.text(1500, ax2.get_ylim()[0], " production default", fontsize=8, color="gray")
    ax2.set_xlabel("sweeps"); ax2.set_ylabel(r"$\varepsilon_c$", color="tab:blue")
    ax3 = ax2.twinx()
    ax3.semilogx(SWEEPS, summary["rmse_vs_ref"], "s--", color="tab:red",
                 label="RMSE vs 6000-sweep reference")
    ax3.set_ylabel(r"RMSE of $m(\varepsilon)$", color="tab:red")
    ax2.set_title("convergence of the estimate with run length")
    h2, l2 = ax2.get_legend_handles_labels(); h3, l3 = ax3.get_legend_handles_labels()
    ax2.legend(h2 + h3, l2 + l3, fontsize=8, loc="center right")
    fig.suptitle("S4: simulation length at the baseline point", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "sens_equilibration.png"), dpi=130)
    print("Saved sens_equilibration.png")


if __name__ == "__main__":
    main()
