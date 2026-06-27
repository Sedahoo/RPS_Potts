"""
Phase 1, step 2: sweep epsilon and plot m_psi(epsilon), one curve per average
degree k. The drop from ~1 to ~0 is the phase transition; larger k pushes it to
higher epsilon (connectivity stabilises order), because every energy is U=k*(...)
so a bigger k acts like a lower effective temperature T/k.
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.meanfield import hmf_run
from common.observables import order_parameter


def sweep(eps_values, k, T=0.65, steps=4000):
    return np.array([order_parameter(hmf_run(e, k=k, T=T, steps=steps))
                     for e in eps_values])


def main():
    eps_values = np.linspace(0.0, 1.0, 51)
    k_values = [2, 5, 10, 50, 200]
    plt.figure(figsize=(8, 5.5))
    for k in k_values:
        m = sweep(eps_values, k)
        plt.plot(eps_values, m, marker="o", ms=3, label=f"k = {k}")
        below = np.where(m < 0.5)[0]
        eps_c = eps_values[below[0]] if len(below) else float("nan")
        print(f"  k={k:>3}:  transition near epsilon ~ {eps_c:.2f}")
    plt.axhline(0.5, color="gray", ls="--", lw=1)
    plt.xlabel("epsilon  (cyclic-dominance strength)")
    plt.ylabel(r"$m_\psi$  (order parameter)")
    plt.title("Phase 1: order -> disorder transition in HMF\n"
              "higher connectivity k pushes the transition to higher epsilon")
    plt.ylim(-0.02, 1.05); plt.legend(title="avg degree")
    plt.tight_layout()
    plt.savefig("hmf_sweep.png", dpi=130)
    print("Saved hmf_sweep.png")


if __name__ == "__main__":
    main()
