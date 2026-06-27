"""
Phase 1: Homogeneous Mean-Field (HMF) model of Potts-RPS dynamics.

The simplest version: no network, no agents -- just 3 coupled equations for the
fractions playing Rock/Paper/Scissors in a well-mixed population. Low epsilon ->
the population ORDERS; high epsilon -> it CYCLES forever. See common/meanfield.py
for the engine and common/observables.py for the m_psi order parameter.
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.meanfield import hmf_run
from common.observables import order_parameter, psi_series


def main():
    cases = [("Low epsilon (ordering)", 0.2), ("High epsilon (cycling)", 0.9)]
    fig, axes = plt.subplots(2, len(cases), figsize=(11, 7), sharex=True)
    for col, (title, eps) in enumerate(cases):
        hist = hmf_run(eps)
        m_psi = order_parameter(hist)
        psi_mag = np.abs(psi_series(hist))

        ax = axes[0, col]
        ax.plot(hist[:, 0], label="rock",     color="tab:red")
        ax.plot(hist[:, 1], label="paper",    color="tab:green")
        ax.plot(hist[:, 2], label="scissors", color="tab:blue")
        ax.set_title(f"{title}\n(eps={eps})"); ax.set_ylabel("fraction")
        ax.set_ylim(-0.02, 1.02); ax.legend(loc="upper right", fontsize=8)

        ax = axes[1, col]
        ax.plot(psi_mag, color="black", lw=1)
        ax.set_title(f"m_psi = {m_psi:.3f}  ->  "
                     f"{'ORDERED' if m_psi > 0.5 else 'DISORDERED'}")
        ax.set_ylabel("|psi(t)|"); ax.set_xlabel("time step"); ax.set_ylim(-0.02, 1.02)
        print(f"  eps={eps}:  m_psi={m_psi:.3f}")

    fig.suptitle("Phase 1: Homogeneous Mean-Field Potts-RPS  (k=10, T=0.65)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig("hmf_phase1.png", dpi=130)
    print("Saved hmf_phase1.png")


if __name__ == "__main__":
    main()
