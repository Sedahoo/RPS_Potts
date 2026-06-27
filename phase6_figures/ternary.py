"""
Phase 6a: the RPS state plotted on a TERNARY simplex.

Every population state (r, p, s) with r+p+s=1 is a point inside a triangle whose
corners are pure Rock / Paper / Scissors. Watching the trajectory move in this
triangle is the most intuitive picture of the two phases:
  * Low eps  -> the path spirals INTO a corner (consensus / ordered).
  * High eps -> the path ORBITS the centre (cyclic coexistence).
This is the original repo's TERNARY experiment, in a few lines.
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase2_mc"))
from mc import build_graph, run_mc

# triangle corners: Rock bottom-left, Paper bottom-right, Scissors top
CORNERS = {"R": (0.0, 0.0), "P": (1.0, 0.0), "S": (0.5, np.sqrt(3) / 2)}


def to_xy(fr):
    """Map (r, p, s) rows to (x, y) inside the triangle."""
    r, p, s = fr[:, 0], fr[:, 1], fr[:, 2]
    x = p * CORNERS["P"][0] + s * CORNERS["S"][0]
    y = s * CORNERS["S"][1]
    return x, y


def draw_triangle(ax):
    pts = [CORNERS["R"], CORNERS["P"], CORNERS["S"], CORNERS["R"]]
    ax.plot([a for a, _ in pts], [b for _, b in pts], color="black", lw=1)
    for name, (x, y) in CORNERS.items():
        ax.annotate({"R": "Rock", "P": "Paper", "S": "Scissors"}[name],
                    (x, y), ha="center",
                    va="bottom" if name == "S" else "top", fontweight="bold")
    ax.set_aspect("equal"); ax.axis("off")


def main():
    neighbors = build_graph("ER", 500, 10, seed=1)
    cases = [("Low eps (spirals to a corner)", 0.2),
             ("High eps (orbits the centre)", 0.95)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, (title, eps) in zip(axes, cases):
        fr = run_mc(neighbors, eps, T=0.65, sweeps=1200, seed=3)
        x, y = to_xy(fr)
        draw_triangle(ax)
        ax.scatter(x, y, c=np.arange(len(x)), cmap="viridis", s=4)
        ax.plot(x, y, color="gray", lw=0.3, alpha=0.5)
        ax.set_title(f"{title}\n(eps={eps})")
    fig.suptitle("Phase 6a: RPS dynamics on the ternary simplex "
                 "(colour = time)", fontweight="bold")
    fig.tight_layout()
    fig.savefig("ternary.png", dpi=130)
    print("Saved ternary.png")


if __name__ == "__main__":
    main()
