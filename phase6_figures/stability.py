"""
Phase 6c: stability of the symmetric fixed point.

The fully-mixed state (1/3, 1/3, 1/3) is always a fixed point of the mean-field
dynamics. The transition is really a question about ITS STABILITY: nudge the
system slightly away from it and see what happens.
  * Low eps  -> the perturbation grows and the system locks onto a CORNER
                (consensus): the mixed state is unstable toward ordering.
  * High eps -> the perturbation grows into a sustained OSCILLATION (limit
                cycle): unstable toward cycling.
This deterministic view (HMF) is the analogue of the original STABILITY_ANALYSIS
experiment, which perturbs the fixed point and watches the resulting signal.
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase1_hmf"))
from hmf import run

DELTA = 0.02  # size of the nudge away from (1/3, 1/3, 1/3)


def main():
    eps_list = [0.2, 0.5, 0.7, 0.95]
    init = (1/3 + DELTA, 1/3, 1/3 - DELTA)   # small perturbation, still sums to 1
    fig, axes = plt.subplots(1, len(eps_list), figsize=(14, 3.6), sharey=True)
    for ax, eps in zip(axes, eps_list):
        hist = run(eps, k=10, T=0.65, steps=600, init=init)
        ax.plot(hist[:, 0], color="tab:red",   lw=0.9, label="rock")
        ax.plot(hist[:, 1], color="tab:green", lw=0.9, label="paper")
        ax.plot(hist[:, 2], color="tab:blue",  lw=0.9, label="scissors")
        ax.set_title(f"eps = {eps}")
        ax.set_xlabel("time step"); ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("fraction"); axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Phase 6c: nudge the mixed fixed point by delta={DELTA} -> "
                 "consensus (low eps) vs limit cycle (high eps)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig("stability.png", dpi=130)
    print("Saved stability.png")


if __name__ == "__main__":
    main()
