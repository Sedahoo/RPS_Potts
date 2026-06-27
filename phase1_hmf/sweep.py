"""
Phase 1, step 2: sweep epsilon and plot the order parameter m_psi(epsilon).

In Phase 1 we looked at two single values of epsilon. Here we scan the whole
range 0 -> 1 and watch m_psi fall from ~1 (ordered) to ~0 (cycling). That drop
IS the phase transition, seen as a curve.

Bonus: we draw one curve per average degree k. This previews the project's
main claim -- "connectivity stabilises order". In the mean field, every energy
is U = k * (...), so a larger k makes the Glauber rule steeper (effectively a
lower temperature T/k). Ordering therefore survives to higher epsilon as k
grows, and the transition curve slides to the right.
"""

import numpy as np
import matplotlib.pyplot as plt

from hmf import run, order_parameter   # reuse the Phase 1 engine


def sweep(eps_values, k, T=0.65, steps=4000):
    """Return m_psi for each epsilon at fixed k."""
    out = np.empty(len(eps_values))
    for i, eps in enumerate(eps_values):
        m_psi, _ = order_parameter(run(eps, k=k, T=T, steps=steps))
        out[i] = m_psi
    return out


def main():
    eps_values = np.linspace(0.0, 1.0, 51)
    k_values = [2, 5, 10, 50, 200]

    plt.figure(figsize=(8, 5.5))
    for k in k_values:
        m = sweep(eps_values, k)
        plt.plot(eps_values, m, marker="o", ms=3, label=f"k = {k}")

    plt.axhline(0.5, color="gray", ls="--", lw=1)
    plt.xlabel("epsilon  (cyclic-dominance strength)")
    plt.ylabel(r"$m_\psi$  (order parameter)")
    plt.title("Phase 1: order -> disorder transition in HMF\n"
              "higher connectivity k pushes the transition to higher epsilon")
    plt.ylim(-0.02, 1.05)
    plt.legend(title="avg degree")
    plt.tight_layout()
    out = "hmf_sweep.png"
    plt.savefig(out, dpi=130)
    print(f"Saved {out}")

    # print the approximate transition point (where m_psi crosses 0.5) per k
    for k in k_values:
        m = sweep(eps_values, k)
        below = np.where(m < 0.5)[0]
        eps_c = eps_values[below[0]] if len(below) else float("nan")
        print(f"  k={k:>3}:  transition near epsilon ~ {eps_c:.2f}")


if __name__ == "__main__":
    main()
