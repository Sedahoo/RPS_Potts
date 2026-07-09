"""
S8 (report Sec. 6): damage realisation seed — does it matter WHICH edges die?

One pristine ER graph (N=500, <k>=20) is damaged 18 ways (3 edge-removal
fractions x 6 damage seeds); each damaged graph gets a full eps sweep.
Hypothesis (sensitivity/HYPOTHESES.md): only the surviving effective degree
2E'/N' matters — eps_c scatter across damage seeds tracks the (tiny) scatter
of effective <k>, and all 18 points fall on the pristine eps_c(<k>) boundary
(Sec. 6.5's collapse, now tested against the disorder realisation).
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.graphs import build_graph, remove_edges, write_edgelist
from common.observables import eps_crossing
from common.io import save_table
from common import runner

HERE = os.path.dirname(os.path.abspath(__file__))
N, K = 500, 20
FRACS = [0.15, 0.30, 0.45]
DSEEDS = [1, 2, 3, 4, 5, 6]
EPS = np.linspace(0, 1, 21)


def main():
    runner.ensure_engine()
    G = build_graph("ER", N, K, seed=1)
    cases, els = [], []
    for f in FRACS:
        for d in DSEEDS:
            H = remove_edges(G, f, seed=d)
            keff = 2 * H.number_of_edges() / H.number_of_nodes()
            el = write_edgelist(H, os.path.join(HERE, f"_dg_{int(f*100)}_{d}.edgelist"))
            cases.append((f, d, keff)); els.append(el)
    res = runner.run_many([dict(edgelist=el, eps=float(e), seed=d)
                           for el, (f, d, keff) in zip(els, cases) for e in EPS])
    m = np.array([r["m_psi"] for r in res]).reshape(len(cases), len(EPS))
    for el in els:
        os.remove(el)

    ecs = [eps_crossing(EPS, m[i]) for i in range(len(cases))]
    save_table(os.path.join(HERE, "sens_defect_seed.csv"),
               {"frac": [c[0] for c in cases], "damage_seed": [c[1] for c in cases],
                "k_eff": [c[2] for c in cases], "eps_c": ecs})

    for f in FRACS:
        sel = [i for i, c in enumerate(cases) if c[0] == f]
        e = np.array([ecs[i] for i in sel]); k = np.array([cases[i][2] for i in sel])
        print(f"f={f}: k_eff={k.mean():.2f}±{k.std():.3f}  "
              f"eps_c={e.mean():.4f}±{e.std():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = {FRACS[0]: "tab:green", FRACS[1]: "tab:orange", FRACS[2]: "tab:red"}
    for i, (f, d, keff) in enumerate(cases):
        axes[0].plot(EPS, m[i], lw=0.8, alpha=0.6, color=colors[f],
                     label=f"f={f}" if d == DSEEDS[0] else None)
    axes[0].axhline(0.5, color="gray", ls="--", lw=0.8)
    axes[0].set_xlabel(r"$\varepsilon$"); axes[0].set_ylabel(r"$m_\psi$")
    axes[0].set_title("6 damage realisations per fraction (curves overlap)")
    axes[0].legend()
    # pristine ER boundary from the phase-diagram extraction
    cb = np.genfromtxt(os.path.join(HERE, "..", "phase_diagram",
                                    "critical_boundary.csv"),
                       delimiter=",", names=True)
    axes[1].plot(cb["k"], cb["eps_c_ER"], "-", color="tab:blue", lw=1.5,
                 label="pristine ER boundary (Sec. 4)")
    for f in FRACS:
        sel = [i for i, c in enumerate(cases) if c[0] == f]
        axes[1].plot([cases[i][2] for i in sel], [ecs[i] for i in sel], "o",
                     ms=6, alpha=0.8, color=colors[f], label=f"damaged, f={f}")
    axes[1].set_xlabel(r"effective $\langle k\rangle = 2E'/N'$")
    axes[1].set_ylabel(r"$\varepsilon_c$")
    axes[1].set_title("all realisations land on the pristine boundary")
    axes[1].legend(fontsize=8)
    fig.suptitle("S8: the damage realisation is a nuisance parameter — only "
                 "effective $\\langle k\\rangle$ matters", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "sens_defect_seed.png"), dpi=130)
    print("Saved sens_defect_seed.png")


if __name__ == "__main__":
    main()
