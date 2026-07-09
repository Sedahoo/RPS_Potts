"""
Sec. 4 across the parameter space: four additional (<k> x eps) phase diagrams
around the headline one (ER, N=800, T=0.65) -- two temperatures either side
of the production value and two system sizes either side of the production
size -- each with its extracted boundary, overlaid on the reference boundary.

Hypotheses (embedded in the report): the two-phase structure is universal
across the grid; T slides the whole boundary down (hotter) / up (colder) with
the shift largest at small <k> (effective noise ~T/k, cf. Sec. 4.2); N barely
moves the boundary at this resolution but shifts it slightly down at larger N
(the 1/N first-order drift of Sec. 5.1).
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
DIAGRAMS = [dict(T=0.30, N=800), dict(T=1.00, N=800),
            dict(T=0.65, N=300), dict(T=0.65, N=2000)]
DEGREES = [2, 4, 6, 8, 10, 14, 18, 22, 26, 30, 35, 40]
EPS = np.linspace(0.0, 1.0, 21)
SEED = 1


def main():
    runner.ensure_engine()
    ref = np.genfromtxt(os.path.join(HERE, "critical_boundary.csv"),
                        delimiter=",", names=True)
    long = {k_: [] for k_ in ("T", "N", "degree", "epsilon", "m_psi")}
    bnd = {k_: [] for k_ in ("T", "N", "degree", "eps_c")}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for ax, dg in zip(axes.flat, DIAGRAMS):
        els = {k: write_edgelist(build_graph("ER", dg["N"], k, seed=SEED),
                                 os.path.join(HERE, f"_xg_{k}.edgelist"))
               for k in DEGREES}
        res = runner.run_many([dict(edgelist=els[k], eps=float(e), temp=dg["T"],
                                    seed=SEED) for k in DEGREES for e in EPS])
        m = np.array([r["m_psi"] for r in res]).reshape(len(DEGREES), len(EPS))
        for el in els.values():
            os.remove(el)
        ecs = [eps_crossing(EPS, m[i]) for i in range(len(DEGREES))]
        for i, k in enumerate(DEGREES):
            for j, e in enumerate(EPS):
                long["T"].append(dg["T"]); long["N"].append(dg["N"])
                long["degree"].append(k); long["epsilon"].append(e)
                long["m_psi"].append(m[i, j])
            bnd["T"].append(dg["T"]); bnd["N"].append(dg["N"])
            bnd["degree"].append(k); bnd["eps_c"].append(ecs[i])

        im = ax.pcolormesh(DEGREES, EPS, m.T, cmap="RdBu_r", vmin=0, vmax=1,
                           shading="nearest")
        ax.plot(DEGREES, ecs, "k-o", ms=4, lw=1.5, label="boundary (this diagram)")
        ax.plot(ref["k"], ref["eps_c_ER"], "--", color="lime", lw=1.8,
                label="reference ($T{=}0.65$, $N{=}800$)")
        ax.set_title(f"$T{{=}}{dg['T']:g}$, $N{{=}}{dg['N']}$")
        ax.legend(fontsize=8, loc="upper left")
    for ax in axes[1]:
        ax.set_xlabel(r"average degree $\langle k\rangle$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\varepsilon$")
    fig.colorbar(im, ax=axes, label=r"$m_\psi$", shrink=0.85)
    fig.suptitle("Extra phase diagrams (ER): two temperatures, two system "
                 "sizes; the ordered/cycling structure is universal, only the "
                 "boundary moves", fontweight="bold")
    fig.savefig(os.path.join(HERE, "extra_diagrams.png"), dpi=130,
                bbox_inches="tight")
    print("Saved extra_diagrams.png")
    save_table(os.path.join(HERE, "extra_diagrams.csv"), long)
    save_table(os.path.join(HERE, "extra_diagrams_boundary.csv"), bnd)

    b = {k_: np.array(v) for k_, v in bnd.items()}
    for dg in DIAGRAMS:
        sel = (b["T"] == dg["T"]) & (b["N"] == dg["N"])
        refi = np.interp(b["degree"][sel], ref["k"], ref["eps_c_ER"])
        dvi = b["eps_c"][sel] - refi
        print(f"T={dg['T']:g} N={dg['N']}: mean boundary shift vs reference "
              f"{dvi.mean():+.3f} (min {dvi.min():+.3f}, max {dvi.max():+.3f})")


if __name__ == "__main__":
    main()
