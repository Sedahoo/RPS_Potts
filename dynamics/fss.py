"""
Phase 6b: Finite-Size Scaling. The m_psi(eps) curve is rounded on small networks
and sharpens as N grows, converging to the true critical point. Uses the shared
C++ engine for speed.
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.graphs import build_graph, write_edgelist
from common import runner


def mc_curve(N, k, eps_vals, T=0.65, sweeps=1200, seed=1):
    G = build_graph("ER", N, k, seed=seed)
    edgelist = write_edgelist(G, os.path.join(os.path.dirname(__file__), f"_fss_N{N}.edgelist"))
    out = np.array([runner.run_engine(edgelist, e, temp=T, sweeps=sweeps, seed=seed)["m_psi"]
                    for e in eps_vals])
    os.remove(edgelist)
    return out


def main():
    runner.ensure_engine()
    k = 10
    eps_vals = np.linspace(0.35, 0.75, 21)
    plt.figure(figsize=(8, 5.5))
    for N in [200, 500, 1000, 2000]:
        plt.plot(eps_vals, mc_curve(N, k, eps_vals), "o-", ms=3, label=f"N = {N}")
        print(f"  N={N:>4} done")
    plt.axhline(0.5, color="gray", ls="--", lw=1)
    plt.xlabel("epsilon"); plt.ylabel(r"$m_\psi$")
    plt.title(f"Phase 6b: finite-size scaling (ER, <k>={k}, T=0.65)\n"
              "the transition sharpens as N grows")
    plt.ylim(-0.02, 1.05); plt.legend(title="system size"); plt.tight_layout()
    plt.savefig("fss.png", dpi=130)
    print("Saved fss.png")


if __name__ == "__main__":
    main()
