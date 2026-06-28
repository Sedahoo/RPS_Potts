"""
Phase 8: DEFECTS (quenched disorder in the network).

The thesis showed connectivity stabilises order. Defects do the opposite: we
damage the network -- remove a fraction f of EDGES (disabled links) or NODES
(vacancies) -- and watch the ordered phase shrink.

For each defect fraction f we sweep epsilon and plot m_psi(eps). The transition
should slide to LOWER epsilon as f grows: a damaged, sparser network can sustain
order against weaker cyclic pressure. This is the network analogue of how
real-world disorder erodes collective ordering.
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.graphs import build_graph, remove_edges, remove_nodes, write_edgelist
from common import runner

N, K = 1000, 20          # start dense so there is room to remove
SEEDS = list(range(1, 7))
HERE = os.path.dirname(__file__)


def curve(damage_fn, frac, eps_vals):
    """m_psi(eps) averaged over seeds, after applying damage_fn(G, frac, seed)."""
    jobs, shapes = [], []
    for s in SEEDS:
        G = damage_fn(build_graph("ER", N, K, seed=s), frac, seed=s)
        el = write_edgelist(G, os.path.join(HERE, f"_dfx_{s}.edgelist"))
        shapes.append((el, G.number_of_nodes(),
                       2 * G.number_of_edges() / G.number_of_nodes()))
        for e in eps_vals:
            jobs.append(dict(edgelist=el, eps=float(e), sweeps=1000, seed=s))
    res = runner.run_many(jobs)
    m = np.array([r["m_psi"] for r in res]).reshape(len(SEEDS), len(eps_vals))
    for el, _, _ in shapes:
        os.remove(el)
    mean_k = np.mean([k for _, _, k in shapes])
    return m.mean(axis=0), mean_k


def main():
    runner.ensure_engine()
    eps_vals = np.linspace(0.0, 1.0, 26)
    fracs = [0.0, 0.3, 0.6, 0.8]
    panels = [("Edge defects (disabled links)", remove_edges),
              ("Node defects (vacancies)", remove_nodes)]

    rows = []   # long-format CSV: defect_type(0=edge,1=node), f, epsilon, m_psi, mean_k
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for type_code, (ax, (title, fn)) in enumerate(zip(axes, panels)):
        for f in fracs:
            m, mean_k = curve(fn, f, eps_vals)
            ax.plot(eps_vals, m, "o-", ms=3,
                    label=f"f={f:.1f}  (<k>~{mean_k:.0f})")
            for e, mv in zip(eps_vals, m):
                rows.append((type_code, f, e, mv, mean_k))
        ax.axhline(0.5, color="gray", ls="--", lw=1)
        ax.set_title(title); ax.set_xlabel("epsilon"); ax.set_ylim(-0.02, 1.05)
        ax.legend(title="defect fraction", fontsize=8)
    axes[0].set_ylabel(r"$m_\psi$")
    fig.suptitle("Phase 8: defects erode the ordered phase "
                 "(transition slides to lower epsilon)", fontweight="bold")
    fig.tight_layout()
    fig.savefig("defects.png", dpi=130)
    print("Saved defects.png")
    np.savetxt("defects.csv", np.array(rows), delimiter=",",
               header="defect_type_0edge_1node,f,epsilon,m_psi,mean_k", comments="")
    print("Saved defects.csv")


if __name__ == "__main__":
    main()
