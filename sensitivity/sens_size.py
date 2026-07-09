"""
S3 (report Sec. 5): system size N at the k=20 operating point.

Extends the report's FSS section (k=10) to the baseline degree: full eps
sweeps at six sizes, 4 graph seeds each. Hypothesis
(sensitivity/HYPOTHESES.md): eps_c settles by N~1000 (it is a property of
<k>, not N); the transition width shrinks with N; seed scatter and
cycling-phase m_psi both shrink ~1/sqrt(N).
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
NS = [125, 250, 500, 1000, 2000, 4000]
SEEDS = [1, 2, 3, 4]
EPS = np.linspace(0, 1, 21)              # coarse full sweep (curves, cycling metrics)
EPS_ZOOM = np.linspace(0.45, 0.80, 29)   # step 0.0125 across the transition
K = 20


def main():
    runner.ensure_engine()
    els = {(n, s): write_edgelist(build_graph("ER", n, K, seed=s),
                                  os.path.join(HERE, f"_ng_{n}_{s}.edgelist"))
           for n in NS for s in SEEDS}
    points = [(n, e, s) for n in NS for e in EPS for s in SEEDS]
    zpoints = [(n, e, s) for n in NS for e in EPS_ZOOM for s in SEEDS]
    res = runner.run_many([dict(edgelist=els[(n, s)], eps=float(e), seed=s)
                           for n, e, s in points])
    zres = runner.run_many([dict(edgelist=els[(n, s)], eps=float(e), seed=s)
                            for n, e, s in zpoints])
    m = np.array([r["m_psi"] for r in res]).reshape(len(NS), len(EPS), len(SEEDS))
    mz = np.array([r["m_psi"] for r in zres]).reshape(len(NS), len(EPS_ZOOM), len(SEEDS))
    for el in els.values():
        os.remove(el)

    summary = {"N": [], "eps_c": [], "eps_c_std": [], "width": [],
               "m_cycle": [], "sd_cycle": []}
    curves = {"epsilon": EPS}
    zcurves = {"epsilon": EPS_ZOOM}
    cyc = EPS >= 0.8                                    # deep cycling region
    for i, n in enumerate(NS):
        mean_curve = m[i].mean(axis=1)
        zmean = mz[i].mean(axis=1)
        # eps_c and width from the fine zoom grid (the coarse grid quantises them)
        per_seed = [eps_crossing(EPS_ZOOM, mz[i, :, s]) for s in range(len(SEEDS))]
        summary["N"].append(n)
        summary["eps_c"].append(np.mean(per_seed))
        summary["eps_c_std"].append(np.std(per_seed))
        summary["width"].append(eps_crossing(EPS_ZOOM, zmean, 0.25)
                                - eps_crossing(EPS_ZOOM, zmean, 0.75))
        summary["m_cycle"].append(mean_curve[cyc].mean())
        summary["sd_cycle"].append(m[i][cyc].std())
        curves[f"m_N{n}"] = mean_curve
        zcurves[f"m_N{n}"] = zmean
    save_table(os.path.join(HERE, "sens_size.csv"), summary)
    save_table(os.path.join(HERE, "sens_size_curves.csv"), curves)
    save_table(os.path.join(HERE, "sens_size_zoom.csv"), zcurves)

    for i, n in enumerate(NS):
        print(f"N={n:5d}: eps_c={summary['eps_c'][i]:.4f}±{summary['eps_c_std'][i]:.4f}"
              f"  width={summary['width'][i]:.4f}  m_cycle={summary['m_cycle'][i]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(NS)))
    for i, n in enumerate(NS):
        axes[0].plot(EPS_ZOOM, mz[i].mean(axis=1), "o-", ms=3, color=cmap[i],
                     label=f"N={n}")
    axes[0].axhline(0.5, color="gray", ls="--", lw=0.8)
    axes[0].set_xlabel(r"$\varepsilon$"); axes[0].set_ylabel(r"$m_\psi$")
    axes[0].set_title("zoom on the transition (step 0.0125)")
    axes[0].legend(fontsize=8)
    ns = np.array(NS, dtype=float)
    axes[1].loglog(ns, summary["width"], "o-", label="transition width (0.75-0.25)")
    axes[1].loglog(ns, summary["m_cycle"], "s-", label=r"$m_\psi$ in cycling phase")
    axes[1].loglog(ns, summary["sd_cycle"], "^-", label="seed scatter (cycling)")
    ref = summary["m_cycle"][0] * np.sqrt(ns[0] / ns)
    axes[1].loglog(ns, ref, "k:", lw=1, label=r"$\propto 1/\sqrt{N}$")
    axes[1].set_xlabel("N"); axes[1].set_ylabel("value")
    axes[1].set_title("finite-size effects scale away"); axes[1].legend(fontsize=8)
    fig.suptitle("S3: system size at $\\langle k\\rangle=20$", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "sens_size.png"), dpi=130)
    print("Saved sens_size.png")


if __name__ == "__main__":
    main()
