"""
Phase 2, validation: overlay the stochastic MC against the deterministic HMF.

We sweep epsilon and plot m_psi(epsilon) for both models on the same axes.
The lesson: they agree deep in each phase, but the MC curve is NOISY and
SMEARED near the transition, while HMF is razor-sharp. That smearing is real
finite-size physics -- a 500-node network is not an infinite well-mixed soup.
This gap is exactly why the project keeps all three models around.

Note the runtime: each MC point is a full simulation. Doing this in pure
Python already feels slow -- that is our cue to port the engine to C++ (Phase 3).
"""

import sys, os, time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase1_hmf"))
from hmf import run as hmf_run, order_parameter as hmf_order   # Phase 1 engine
from mc import build_graph, run_mc, order_parameter as mc_order  # Phase 2 engine


def main():
    k, T = 10, 0.65
    eps_values = np.linspace(0.0, 1.0, 26)

    # HMF curve (fast, deterministic)
    hmf = np.array([hmf_order(hmf_run(e, k=k, T=T))[0] for e in eps_values])

    # MC curve (slow, stochastic) -- one fixed ER graph, averaged over a few seeds
    neighbors = build_graph("ER", 500, k, seed=1)
    t0 = time.time()
    mc = np.empty(len(eps_values))
    for i, e in enumerate(eps_values):
        vals = [mc_order(run_mc(neighbors, e, T=T, sweeps=1200, seed=s)) for s in (1, 2, 3)]
        mc[i] = np.mean(vals)
    print(f"MC sweep took {time.time()-t0:.1f}s for {len(eps_values)} points x 3 seeds")

    plt.figure(figsize=(8, 5.5))
    plt.plot(eps_values, hmf, "-", color="tab:orange", lw=2, label="HMF (mean field)")
    plt.plot(eps_values, mc, "o-", color="tab:blue", ms=4, label="MC (N=500 ER, ground truth)")
    plt.axhline(0.5, color="gray", ls="--", lw=1)
    plt.xlabel("epsilon"); plt.ylabel(r"$m_\psi$")
    plt.title(f"MC vs HMF: agreement in the bulk, divergence at the transition\n"
              f"(<k>={k}, T={T})")
    plt.ylim(-0.02, 1.05); plt.legend()
    plt.tight_layout()
    plt.savefig("mc_vs_hmf.png", dpi=130)
    print("Saved mc_vs_hmf.png")


if __name__ == "__main__":
    main()
