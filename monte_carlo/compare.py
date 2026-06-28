"""
Phase 2, validation: overlay the stochastic MC against the deterministic HMF.
They agree deep in each phase but the MC curve is rounded/shifted near the
transition -- finite-size physics the mean field can't see.
"""

import os, sys, time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.graphs import build_graph, neighbor_lists
from common.mc_python import run_mc
from common.meanfield import hmf_run
from common.observables import order_parameter
from common.io import save_table


def main():
    k, T = 10, 0.65
    eps_values = np.linspace(0.0, 1.0, 26)
    hmf = np.array([order_parameter(hmf_run(e, k=k, T=T)) for e in eps_values])

    neighbors = neighbor_lists(build_graph("ER", 500, k, seed=1))
    t0 = time.time()
    mc = np.array([np.mean([order_parameter(run_mc(neighbors, e, T=T, sweeps=1200, seed=s),
                                            burn_in_frac=0.3) for s in (1, 2, 3)])
                   for e in eps_values])
    print(f"MC sweep took {time.time()-t0:.1f}s")

    plt.figure(figsize=(8, 5.5))
    plt.plot(eps_values, hmf, "-", color="tab:orange", lw=2, label="HMF (mean field)")
    plt.plot(eps_values, mc, "o-", color="tab:blue", ms=4,
             label="MC (N=500 ER, ground truth)")
    plt.axhline(0.5, color="gray", ls="--", lw=1)
    plt.xlabel("epsilon"); plt.ylabel(r"$m_\psi$")
    plt.title(f"MC vs HMF: agreement in the bulk, divergence at the transition\n"
              f"(<k>={k}, T={T})")
    plt.ylim(-0.02, 1.05); plt.legend(); plt.tight_layout()
    plt.savefig("mc_vs_hmf.png", dpi=130)
    print("Saved mc_vs_hmf.png")
    save_table("mc_vs_hmf.csv", {"epsilon": eps_values, "hmf": hmf, "mc": mc})


if __name__ == "__main__":
    main()
