"""
Sec. 3.3 across the parameter space: the direct MC vs HMF overlay of
compare.py repeated over T, <k> and N (one axis at a time around ER,
N=500, <k>=10, T=0.65), plus a BA row (BA overlay at k=10 and k=20, and a
direct ER-vs-BA MC comparison).

Unlike compare.py (pure-Python MC, kept as the teaching reference), the MC
here is the validated C++ engine (Sec. 2 / 2.1), 3 seeds averaged, which is
what makes a 12-panel grid affordable.

Hypotheses (embedded in the report): the mean field misplaces the transition
in a direction and amount that varies systematically -- the MC-HMF gap in
eps_c widens with T (MC boundary falls, HMF nearly T-blind at these k),
narrows with k (toward the high-k regime of Sec. 4.2 where it reverses), and
is N-independent except for MC sharpening. The BA overlays should look like
the ER ones at matched <k> (Sec. 4.1: ER ~ BA), with the HMF fed the same
measured <k>.
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.graphs import build_graph, write_edgelist, degree_dist
from common.meanfield import hmf_run
from common.observables import order_parameter, eps_crossing
from common.io import save_table
from common import runner

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = dict(graph="ER", T=0.65, k=10, N=500)
EPS = np.linspace(0.0, 1.0, 26)
SEEDS = [1, 2, 3]
ROWS = [("vary $T$", [dict(BASE, T=v) for v in (0.40, 0.65, 1.00)]),
        ("vary $\\langle k\\rangle$", [dict(BASE, k=v) for v in (6, 10, 20)]),
        ("vary $N$", [dict(BASE, N=v) for v in (250, 500, 1000)]),
        ("BA", [dict(BASE, graph="BA"), dict(BASE, graph="BA", k=20), None])]


def sweep(cell):
    """Seed-averaged MC curve + HMF curve (measured <k>) for one cell."""
    mcs, kmeas = [], []
    for s in SEEDS:
        G = build_graph(cell["graph"], cell["N"], cell["k"], seed=s)
        _, mean_deg = degree_dist(G)
        kmeas.append(mean_deg)
        el = write_edgelist(G, os.path.join(HERE, "_ovg.edgelist"))
        res = runner.run_many([dict(edgelist=el, eps=float(e), temp=cell["T"],
                                    seed=s) for e in EPS])
        mcs.append([r["m_psi"] for r in res])
        os.remove(el)
    mc = np.mean(mcs, axis=0)
    hmf = np.array([order_parameter(hmf_run(float(e), k=float(np.mean(kmeas)),
                                            T=cell["T"])) for e in EPS])
    return mc, hmf


def main():
    runner.ensure_engine()
    cols = {"epsilon": EPS}
    fig, axes = plt.subplots(len(ROWS), 3, figsize=(13, 13), sharex=True,
                             sharey=True)
    for r_i, (rlabel, cs) in enumerate(ROWS):
        for c_i, cell in enumerate(cs):
            ax = axes[r_i, c_i]
            if cell is None:                       # ER vs BA head-to-head panel
                er = cols["mc_g0_T65_k10_N500"]
                ba = cols["mc_g1_T65_k10_N500"]
                ax.plot(EPS, er, "o-", ms=3, color="tab:blue", label="MC ER")
                ax.plot(EPS, ba, "s-", ms=3, color="tab:orange", label="MC BA")
                ax.set_title("ER vs BA (MC only, $\\langle k\\rangle{=}10$)",
                             fontsize=9)
                ax.axhline(0.5, color="gray", ls="--", lw=0.7)
                ax.legend(fontsize=8)
                continue
            mc, hmf = sweep(cell)
            g_i = 0 if cell["graph"] == "ER" else 1
            tag = f"g{g_i}_T{int(cell['T']*100)}_k{cell['k']}_N{cell['N']}"
            cols[f"mc_{tag}"] = mc
            cols[f"hmf_{tag}"] = hmf
            ec_mc, ec_h = eps_crossing(EPS, mc), eps_crossing(EPS, hmf)
            ax.plot(EPS, mc, "o-", ms=3, color="black", label="MC (engine)")
            ax.plot(EPS, hmf, "s--", ms=3, color="tab:orange", label="HMF")
            ax.axhline(0.5, color="gray", ls="--", lw=0.7)
            ax.set_title(f"{cell['graph']}, $T{{=}}{cell['T']:g}$, "
                         f"$\\langle k\\rangle{{=}}{cell['k']}$, "
                         f"$N{{=}}{cell['N']}$\n"
                         f"$\\varepsilon_c$: MC {ec_mc:.2f} / HMF {ec_h:.2f}",
                         fontsize=9)
            if r_i == 0 and c_i == 0:
                ax.legend(fontsize=8)
        axes[r_i, 0].set_ylabel(f"{rlabel}\n$m_\\psi$")
    for ax in axes[-1]:
        ax.set_xlabel(r"$\varepsilon$")
    fig.suptitle("Sec. 3.3 across $T$, $\\langle k\\rangle$, $N$ (+ BA): direct "
                 "MC vs HMF overlays, C++ engine, 3 seeds averaged",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "mc_vs_hmf_grid.png"), dpi=130)
    print("Saved mc_vs_hmf_grid.png")
    save_table(os.path.join(HERE, "mc_vs_hmf_grid.csv"), cols)


if __name__ == "__main__":
    main()
