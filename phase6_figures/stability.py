"""
Phase 6c: stability of the symmetric fixed point (1/3,1/3,1/3). Nudge it by delta
and integrate the HMF dynamics: low eps -> collapses to a corner (consensus);
high eps -> blooms into a limit cycle. The transition is really a statement about
this fixed point's stability.
"""

import os, sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.meanfield import hmf_run

DELTA = 0.02


def main():
    eps_list = [0.2, 0.5, 0.7, 0.95]
    init = (1/3 + DELTA, 1/3, 1/3 - DELTA)
    fig, axes = plt.subplots(1, len(eps_list), figsize=(14, 3.6), sharey=True)
    for ax, eps in zip(axes, eps_list):
        hist = hmf_run(eps, k=10, T=0.65, steps=600, init=init)
        ax.plot(hist[:, 0], color="tab:red",   lw=0.9, label="rock")
        ax.plot(hist[:, 1], color="tab:green", lw=0.9, label="paper")
        ax.plot(hist[:, 2], color="tab:blue",  lw=0.9, label="scissors")
        ax.set_title(f"eps = {eps}"); ax.set_xlabel("time step"); ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("fraction"); axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Phase 6c: nudge the mixed fixed point by delta={DELTA} -> "
                 "consensus (low eps) vs limit cycle (high eps)", fontweight="bold")
    fig.tight_layout(); fig.savefig("stability.png", dpi=130)
    print("Saved stability.png")


if __name__ == "__main__":
    main()
