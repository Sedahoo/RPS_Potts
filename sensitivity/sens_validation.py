"""
S0 (report Sec. 2): does C++/Python engine agreement hold across parameters?

The headline validation is 4 eps values on one graph with one seed. Here the
same paired comparison runs over N x eps x seed (45 pairs): same graph and
seed number through both engines (independent RNG streams, so agreement is
statistical). Hypothesis (sensitivity/HYPOTHESES.md): same-side-of-0.5
verdicts everywhere except inside the transition region where m ~ 0.5;
|m_py - m_cpp| grows near the transition and shrinks ~1/sqrt(N); no
systematic offset anywhere away from the transition.
"""

import os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.graphs import build_graph, neighbor_lists, write_edgelist
from common.mc_python import run_mc
from common.observables import order_parameter
from common.io import save_table
from common import runner

HERE = os.path.dirname(os.path.abspath(__file__))
NS = [100, 300, 500]
EPSS = [0.2, 0.35, 0.5, 0.65, 0.8]
SEEDS = [7, 8, 9]
K, T, SWEEPS = 10, 0.65, 1500


def _py_job(args):
    neighbors, eps, seed = args
    return order_parameter(run_mc(neighbors, eps, T=T, sweeps=SWEEPS, seed=seed),
                           burn_in_frac=0.3)


def main():
    runner.ensure_engine()
    graphs = {n: build_graph("ER", n, K, seed=42) for n in NS}
    edgelists = {n: write_edgelist(g, os.path.join(HERE, f"_vg_{n}.edgelist"))
                 for n, g in graphs.items()}
    neigh = {n: neighbor_lists(g) for n, g in graphs.items()}

    points = [(n, e, s) for n in NS for e in EPSS for s in SEEDS]
    with ProcessPoolExecutor() as ex:                       # pure-Python side is slow
        m_py = list(ex.map(_py_job, [(neigh[n], e, s) for n, e, s in points]))
    res = runner.run_many([dict(edgelist=edgelists[n], eps=e, temp=T,
                                sweeps=SWEEPS, seed=s) for n, e, s in points])
    m_cpp = [r["m_psi"] for r in res]
    for el in edgelists.values():
        os.remove(el)

    cols = {"N": [p[0] for p in points], "epsilon": [p[1] for p in points],
            "seed": [p[2] for p in points], "m_py": m_py, "m_cpp": m_cpp}
    save_table(os.path.join(HERE, "sens_validation.csv"), cols)

    py = np.array(m_py); cpp = np.array(m_cpp)
    agree = (py > 0.5) == (cpp > 0.5)
    print(f"verdict: {agree.sum()}/{len(agree)} pairs on the same side of 0.5")
    for n, e, s in np.array(points)[~agree]:
        print(f"  DIFFER at N={n:.0f} eps={e} seed={s:.0f}")
    print(f"max |m_py - m_cpp| = {np.max(np.abs(py - cpp)):.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {100: "tab:blue", 300: "tab:orange", 500: "tab:green"}
    for n in NS:
        sel = np.array([p[0] == n for p in points])
        axes[0].plot(py[sel], cpp[sel], "o", ms=5, alpha=0.8, color=colors[n],
                     label=f"N={n}")
        # per-(N, eps) mean absolute gap across seeds
        gaps = [np.mean(np.abs(py[sel & np.isclose([p[1] for p in points], e)]
                               - cpp[sel & np.isclose([p[1] for p in points], e)]))
                for e in EPSS]
        axes[1].plot(EPSS, gaps, "o-", color=colors[n], label=f"N={n}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].axhline(0.5, color="gray", lw=0.6); axes[0].axvline(0.5, color="gray", lw=0.6)
    axes[0].set_xlabel(r"pure-Python $m_\psi$"); axes[0].set_ylabel(r"C++ $m_\psi$")
    axes[0].set_title("45 paired runs (same graph & seed)"); axes[0].legend()
    axes[1].set_xlabel(r"$\varepsilon$"); axes[1].set_ylabel(r"mean $|m_{py}-m_{cpp}|$")
    axes[1].set_title("gap peaks at the transition, shrinks with $N$"); axes[1].legend()
    fig.suptitle("S0: engine validation across $N\\times\\varepsilon\\times$seed",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "sens_validation.png"), dpi=130)
    print("Saved sens_validation.png")


if __name__ == "__main__":
    main()
