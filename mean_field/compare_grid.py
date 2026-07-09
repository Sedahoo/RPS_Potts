"""
Sec. 3.2 across the parameter space: the MC vs HMF vs DMF accuracy test of
compare_suite.py repeated over T, <k> and N (one axis varied at a time around
the base point N=800, <k>=10, T=0.65), on both ER and BA.

Hypotheses (embedded in the report): DMF <= HMF in RMSE everywhere, with the
larger DMF advantage on BA (only DMF sees the heavy-tailed P(k)). RMSE should
GROW with T at fixed k (the MC boundary slides away from the near-T-blind
mean-field one, widening the mismatch region) and SHRINK with k (MC eps_c
rises toward the mean-field value as fluctuations matter less per
neighbourhood). N is a near-null axis: the mean fields are N-blind and the MC
curve is intensive, so RMSE should move only through the sharpening of the MC
transition.
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.graphs import build_graph, write_edgelist, degree_dist
from common.meanfield import hmf_run, dmf_run
from common.observables import order_parameter, eps_crossing
from common.io import save_table
from common import runner

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = dict(T=0.65, k=10, N=800)
AXES = [("T", [0.40, 0.65, 1.00]), ("k", [6, 10, 20]), ("N", [400, 800, 1600])]
GRAPHS = ["ER", "BA"]
EPS = np.linspace(0.0, 1.0, 26)
SEEDS = [1, 2]                      # MC ground truth averaged over 2 graph seeds


def cells():
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
    long = {k_: [] for k_ in ("graph", "T", "k", "N", "rmse_hmf", "rmse_dmf",
                              "epsc_mc", "epsc_hmf", "epsc_dmf")}
    for g_i, gname in enumerate(GRAPHS):
        for c in cs:
            mcs, hmfs, dmfs = [], [], []
            for s in SEEDS:
                G = build_graph(gname, c["N"], c["k"], seed=s)
                pk, mean_deg = degree_dist(G)
                el = write_edgelist(G, os.path.join(HERE, "_cg.edgelist"))
                jobs = [dict(edgelist=el, eps=float(e), temp=c["T"], seed=s)
                        for e in EPS]
                mcs.append([r["m_psi"] for r in runner.run_many(jobs)])
                os.remove(el)
                hmfs.append([order_parameter(hmf_run(float(e), k=mean_deg,
                                                     T=c["T"])) for e in EPS])
                dmfs.append([order_parameter(dmf_run(pk, float(e), T=c["T"]))
                             for e in EPS])
            mc = np.mean(mcs, axis=0)
            hmf = np.mean(hmfs, axis=0)
            dmf = np.mean(dmfs, axis=0)
            long["graph"].append(g_i)
            long["T"].append(c["T"]); long["k"].append(c["k"]); long["N"].append(c["N"])
            long["rmse_hmf"].append(float(np.sqrt(np.mean((hmf - mc) ** 2))))
            long["rmse_dmf"].append(float(np.sqrt(np.mean((dmf - mc) ** 2))))
            long["epsc_mc"].append(eps_crossing(EPS, mc))
            long["epsc_hmf"].append(eps_crossing(EPS, hmf))
            long["epsc_dmf"].append(eps_crossing(EPS, dmf))
            print(f"{gname} T={c['T']:g} k={c['k']} N={c['N']}: "
                  f"RMSE hmf={long['rmse_hmf'][-1]:.3f} dmf={long['rmse_dmf'][-1]:.3f}"
                  f"  epsc mc={long['epsc_mc'][-1]:.3f} hmf={long['epsc_hmf'][-1]:.3f}")
    save_table(os.path.join(HERE, "compare_grid.csv"), long)

    lg = {k_: np.array(v) for k_, v in long.items()}
    fig, axes = plt.subplots(2, len(AXES), figsize=(13, 7.5), sharey=True)
    for g_i, gname in enumerate(GRAPHS):
        for col, (name, vals) in enumerate(AXES):
            ax = axes[g_i, col]
            xs = np.arange(len(vals))
            h_, d_ = [], []
            for v in vals:
                c = dict(BASE); c[name] = v
                sel = ((lg["graph"] == g_i) & (lg["T"] == c["T"])
                       & (lg["k"] == c["k"]) & (lg["N"] == c["N"]))
                h_.append(lg["rmse_hmf"][sel][0]); d_.append(lg["rmse_dmf"][sel][0])
            ax.bar(xs - 0.18, h_, 0.36, color="tab:orange", label="RMSE(HMF)")
            ax.bar(xs + 0.18, d_, 0.36, color="tab:green", label="RMSE(DMF)")
            ax.set_xticks(xs); ax.set_xticklabels([f"{v:g}" for v in vals])
            ax.set_xlabel(name)
            if col == 0:
                ax.set_ylabel(f"{gname}: RMSE vs MC")
            if g_i == 0 and col == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Sec. 3.2 across $T$, $\\langle k\\rangle$, $N$: mean-field "
                 "accuracy vs the MC ground truth (base point $N{=}800$, "
                 "$\\langle k\\rangle{=}10$, $T{=}0.65$)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "compare_grid.png"), dpi=130)
    print("Saved compare_grid.png")


if __name__ == "__main__":
    main()
