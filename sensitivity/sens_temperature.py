"""
S1 (report Sec. 4): temperature x degree — is the phase diagram robust in T,
and does <k> really dominate?

The whole phase diagram is taken at T=0.65. Here eps_c is measured on a
T x k grid (MC, 4 graph seeds) and compared to the HMF prediction on the same
grid. Hypothesis (sensitivity/HYPOTHESES.md): eps_c decreases monotonically
with T at every k (noise destabilises the ferromagnetic pinning); the payoff
scales with k so the effective noise is ~T/k and the T-dependence weakens as
k grows; HMF overestimates eps_c at every (T, k).
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.graphs import build_graph, write_edgelist
from common.meanfield import hmf_run
from common.observables import order_parameter, eps_crossing
from common.io import save_table
from common import runner

HERE = os.path.dirname(os.path.abspath(__file__))
TS = [0.30, 0.50, 0.65, 0.80, 1.00, 1.30]
KS = [8, 20, 40]
SEEDS = [1, 2, 3, 4]
EPS = np.linspace(0, 1, 21)
N = 500


def main():
    runner.ensure_engine()
    els = {(k, s): write_edgelist(build_graph("ER", N, k, seed=s),
                                  os.path.join(HERE, f"_tg_{k}_{s}.edgelist"))
           for k in KS for s in SEEDS}

    points = [(t, k, e, s) for t in TS for k in KS for e in EPS for s in SEEDS]
    res = runner.run_many([dict(edgelist=els[(k, s)], eps=float(e), temp=t, seed=s)
                           for t, k, e, s in points])
    m = np.array([r["m_psi"] for r in res]).reshape(len(TS), len(KS), len(EPS), len(SEEDS))
    for el in els.values():
        os.remove(el)

    summary = {"T": [], "k": [], "eps_c_mc": [], "eps_c_mc_std": [], "eps_c_hmf": []}
    curves = {"epsilon": EPS}
    for i, t in enumerate(TS):
        for j, k in enumerate(KS):
            per_seed = [eps_crossing(EPS, m[i, j, :, s]) for s in range(len(SEEDS))]
            # the HMF map is deterministic and cheap: use a fine grid so its
            # eps_c is not quantised by the MC grid step
            eps_fine = np.linspace(0, 1, 101)
            m_hmf = [order_parameter(hmf_run(float(e), k=k, T=t)) for e in eps_fine]
            summary["T"].append(t); summary["k"].append(k)
            summary["eps_c_mc"].append(np.mean(per_seed))
            summary["eps_c_mc_std"].append(np.std(per_seed))
            summary["eps_c_hmf"].append(eps_crossing(eps_fine, m_hmf))
            curves[f"m_T{int(t*100)}_k{k}"] = m[i, j].mean(axis=1)
    save_table(os.path.join(HERE, "sens_temperature.csv"), summary)
    save_table(os.path.join(HERE, "sens_temperature_curves.csv"), curves)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    cmap = plt.cm.plasma(np.linspace(0.05, 0.85, len(TS)))
    for ax, (j, k) in zip(axes.flat[:3], enumerate(KS)):
        for i, t in enumerate(TS):
            ax.plot(EPS, m[i, j].mean(axis=1), "o-", ms=3, color=cmap[i],
                    label=f"T={t}")
        ax.axhline(0.5, color="gray", ls="--", lw=0.8)
        ax.set_title(f"$\\langle k\\rangle={k}$")
        ax.set_xlabel(r"$\varepsilon$"); ax.set_ylabel(r"$m_\psi$")
        ax.legend(fontsize=8)
    ax = axes.flat[3]
    su = {kk: np.array(v) for kk, v in summary.items()}
    for j, k in enumerate(KS):
        sel = su["k"] == k
        ax.errorbar(su["T"][sel], su["eps_c_mc"][sel], yerr=su["eps_c_mc_std"][sel],
                    fmt="o-", capsize=3, label=f"MC k={k}")
        ax.plot(su["T"][sel], su["eps_c_hmf"][sel], "D--", ms=4, alpha=0.6,
                label=f"HMF k={k}")
    ax.set_xlabel("T"); ax.set_ylabel(r"$\varepsilon_c$")
    ax.set_title(r"$\varepsilon_c(T)$: flatter at larger $\langle k\rangle$")
    ax.legend(fontsize=8, ncol=2)
    fig.suptitle("S1: temperature $\\times$ degree sensitivity of the phase diagram",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "sens_temperature.png"), dpi=130)
    print("Saved sens_temperature.png")

    for j, k in enumerate(KS):
        sel = su["k"] == k
        drop = su["eps_c_mc"][sel][0] - su["eps_c_mc"][sel][-1]
        print(f"k={k}: eps_c falls {drop:+.3f} from T={TS[0]} to T={TS[-1]}")


if __name__ == "__main__":
    main()
