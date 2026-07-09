"""
S2 (report Sec. 4): graph seed vs MC seed — are single-realisation results
representative?

25 (graph seed, engine seed) combinations run the full eps sweep at the
baseline point (ER, N=500, <k>=20). Hypothesis (sensitivity/HYPOTHESES.md):
both seeds are nuisance parameters — eps_c scatter well under one grid step,
m_psi scatter peaking near eps_c, graph- and MC-seed contributions comparable
(ER at <k>=20 is locally homogeneous).
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
GSEEDS = [1, 2, 3, 4, 5]
MSEEDS = [1, 2, 3, 4, 5]
EPS = np.linspace(0, 1, 21)
N, K = 500, 20


def main():
    runner.ensure_engine()
    els = {g: write_edgelist(build_graph("ER", N, K, seed=g),
                             os.path.join(HERE, f"_sg_{g}.edgelist"))
           for g in GSEEDS}
    points = [(g, ms, e) for g in GSEEDS for ms in MSEEDS for e in EPS]
    res = runner.run_many([dict(edgelist=els[g], eps=float(e), seed=ms)
                           for g, ms, e in points])
    m = np.array([r["m_psi"] for r in res]).reshape(len(GSEEDS), len(MSEEDS), len(EPS))
    for el in els.values():
        os.remove(el)

    ec = np.array([[eps_crossing(EPS, m[i, j]) for j in range(len(MSEEDS))]
                   for i in range(len(GSEEDS))])
    save_table(os.path.join(HERE, "sens_seeds.csv"),
               {"graph_seed": np.repeat(GSEEDS, len(MSEEDS)),
                "mc_seed": np.tile(MSEEDS, len(GSEEDS)),
                "eps_c": ec.ravel()})
    save_table(os.path.join(HERE, "sens_seeds_curves.csv"),
               {"epsilon": EPS,
                "m_mean": m.mean(axis=(0, 1)), "m_std": m.std(axis=(0, 1)),
                "m_min": m.min(axis=(0, 1)), "m_max": m.max(axis=(0, 1))})

    total_sd = ec.std()
    graph_sd = ec.mean(axis=1).std()      # scatter of graph means (MC averaged out)
    mc_sd = ec.mean(axis=0).std()         # scatter of MC means (graphs averaged out)
    print(f"eps_c over 25 combos: mean={ec.mean():.4f}  total std={total_sd:.4f}  "
          f"range={ec.max()-ec.min():.4f}")
    print(f"graph-seed std (MC averaged) = {graph_sd:.4f} | "
          f"MC-seed std (graph averaged) = {mc_sd:.4f}")
    imax = np.argmax(m.std(axis=(0, 1)))
    print(f"m_psi scatter peaks at eps={EPS[imax]:.2f} "
          f"(std={m.std(axis=(0,1))[imax]:.3f}), vs {m.std(axis=(0,1))[0]:.4f} at eps=0")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for i in range(len(GSEEDS)):
        for j in range(len(MSEEDS)):
            axes[0].plot(EPS, m[i, j], color="gray", lw=0.6, alpha=0.5)
    axes[0].plot(EPS, m.mean(axis=(0, 1)), "o-", color="tab:blue", label="mean of 25")
    axes[0].axhline(0.5, color="gray", ls="--", lw=0.8)
    axes[0].set_xlabel(r"$\varepsilon$"); axes[0].set_ylabel(r"$m_\psi$")
    axes[0].set_title("25 (graph, MC)-seed curves"); axes[0].legend()
    for j, ms in enumerate(MSEEDS):
        axes[1].plot(GSEEDS, ec[:, j], "o", ms=5, alpha=0.75, label=f"MC seed {ms}")
    axes[1].axhline(ec.mean(), color="k", lw=0.8)
    axes[1].axhspan(ec.mean() - 0.025, ec.mean() + 0.025, color="tab:blue", alpha=0.12,
                    label=r"$\pm$ half a grid step")
    axes[1].set_xlabel("graph seed"); axes[1].set_ylabel(r"$\varepsilon_c$")
    axes[1].set_xticks(GSEEDS)
    axes[1].set_title(r"$\varepsilon_c$ scatter across all 25 combinations")
    axes[1].legend(fontsize=8)
    fig.suptitle("S2: graph seed vs MC seed at the baseline point", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "sens_seeds.png"), dpi=130)
    print("Saved sens_seeds.png")


if __name__ == "__main__":
    main()
