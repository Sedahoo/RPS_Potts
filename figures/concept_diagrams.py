"""
Conceptual diagrams for the paper: not new experiments, just clearer pictures
of objects the paper already defines mathematically.

  * order_parameter_geometry.png -- the (r,p,s) simplex IS the image of psi:
    placing R,P,S at the cube roots of unity (angles 0, 120, 240 degrees)
    makes the barycentric position of a composition x=(r,p,s) coincide
    exactly with psi(x) = r + p*omega + s*omega^2. Left panel: that
    geometry, the cyclic "beats" relation drawn as arrows around the
    perimeter, one sample composition and its psi-vector decomposition.
    Right panel: two real HMF trajectories (common.meanfield.hmf_run) in
    the same triangle -- one below eps_c spiralling into a corner
    (consensus), one above eps_c orbiting the centre (cycling) -- so the
    same picture carries the paper's central order/cycling distinction.

  * network_evolution.png -- three snapshots of one small ER graph's
    node colours (strategies) as the SAME pure-Python Glauber dynamics
    used elsewhere in the repo (common/mc_python.py's update rule) runs
    forward, illustrating what "the population reaches consensus" means
    at the level of individual agents rather than only in the aggregate
    (r,p,s) plot. Fixed node layout across panels so only colour changes.

Both are deterministic (fixed seeds); this script has no CSV output since
neither figure carries data beyond what the caption states, but the
network-evolution snapshot states are saved as a CSV for inspection anyway,
matching the project convention of pairing every figure with its numbers.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common.meanfield import hmf_run
from common.graphs import build_graph
from common.io import save_table

R_COLOR, P_COLOR, S_COLOR = "#c0392b", "#2980b9", "#27ae60"
SNAMES = ["Rock", "Paper", "Scissors"]
SCOLORS = [R_COLOR, P_COLOR, S_COLOR]

# cube roots of unity: R at angle 0, P at 120 deg, S at 240 deg -- chosen so
# that a composition's barycentric position IN THIS TRIANGLE equals psi(x)
# exactly (psi = r*1 + p*omega + s*omega^2), not just up to relabelling.
VR = np.array([1.0, 0.0])
VP = np.array([np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)])
VS = np.array([np.cos(4 * np.pi / 3), np.sin(4 * np.pi / 3)])


def bary(r, p, s):
    return r * VR + p * VP + s * VS


def draw_triangle(ax, with_arrows=True):
    tri = plt.Polygon([VR, VP, VS], closed=True, fill=False,
                      edgecolor="black", lw=1.2, zorder=3)
    ax.add_patch(tri)
    circ = plt.Circle((0, 0), 1.0, fill=False, ls=":", color="gray", lw=0.8)
    ax.add_patch(circ)
    for v, name, c in zip((VR, VP, VS), SNAMES, SCOLORS):
        ax.scatter(*v, s=90, color=c, zorder=5, edgecolor="black", lw=0.8)
        ax.annotate(name, v * 1.20, ha="center", va="center", fontsize=10,
                   fontweight="bold", color=c)
    if with_arrows:
        pairs = [(VR, VP, "P beats R"), (VP, VS, "S beats P"), (VS, VR, "R beats S")]
        for a, b, label in pairs:
            arr = FancyArrowPatch(a, b, connectionstyle="arc3,rad=0.28",
                                  arrowstyle="-|>", mutation_scale=13,
                                  color="dimgray", lw=1.2, zorder=4,
                                  shrinkA=8, shrinkB=8)
            ax.add_patch(arr)
            mid = (a + b) / 2
            lab_pos = mid / np.linalg.norm(mid) * 1.42
            ax.annotate(label, lab_pos, ha="center", va="center", fontsize=7.8,
                       color="dimgray", style="italic")
    ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.55, 1.55)
    ax.set_aspect("equal"); ax.axis("off")


def panel_geometry(ax):
    draw_triangle(ax, with_arrows=True)
    x = (0.50, 0.32, 0.18)
    pos = bary(*x)
    ax.plot([pos[0], pos[0]], [0, pos[1]], ls="--", color="steelblue", lw=1, zorder=2)
    ax.plot([0, pos[0]], [0, 0], ls="--", color="firebrick", lw=1, zorder=2)
    ax.annotate("", xy=pos, xytext=(0, 0),
               arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6), zorder=6)
    ax.scatter(*pos, s=55, color="black", zorder=7)
    ax.annotate(r"$\mathrm{Re}\,\psi$", (pos[0] / 2, -0.13), color="firebrick",
               fontsize=8, ha="center")
    ax.annotate(r"$\mathrm{Im}\,\psi$", (pos[0] - 0.14, pos[1] / 2), color="steelblue",
               fontsize=8, va="center", ha="right")
    ax.annotate(r"$x=(r,p,s)$", pos, xytext=(pos[0] - 0.05, pos[1] + 0.22),
               fontsize=8.5, ha="center")
    ax.text(0, -1.52, r"$\psi=r+p\,\omega+s\,\omega^2,\quad x=(%.2f,%.2f,%.2f),"
           r"\quad|\psi|=%.2f$" % (x[0], x[1], x[2], np.hypot(*pos)),
           fontsize=9, ha="center")
    ax.set_title("Order parameter as a vector\n(simplex = image of $\\psi$)", fontsize=10)


def panel_trajectories(ax, k=10.0, T=0.65, eps_lo=0.30, eps_hi=0.90, steps=4000):
    draw_triangle(ax, with_arrows=False)
    for eps, color, label in ((eps_lo, "#8e44ad", f"$\\varepsilon={eps_lo:g}$ (orders)"),
                              (eps_hi, "#e67e22", f"$\\varepsilon={eps_hi:g}$ (cycles)")):
        traj = hmf_run(eps, k=k, T=T, steps=steps, init=(0.40, 0.35, 0.25))
        pts = np.array([bary(*row) for row in traj])
        tail = pts[-400:]
        ax.plot(tail[:, 0], tail[:, 1], color=color, lw=1.3, alpha=0.9, label=label)
        ax.scatter(*pts[0], marker="x", color=color, s=35, zorder=6)
    ax.scatter(0, 0, marker="+", color="black", s=50, zorder=6)
    ax.legend(loc="lower center", fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.06))
    ax.set_title(f"HMF trajectories ($\\langle k\\rangle{{=}}{k:g}$, $T{{=}}{T:g}$):\n"
               "corner = consensus, loop = cycling", fontsize=10)


def make_order_parameter_geometry():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6))
    panel_geometry(axes[0])
    panel_trajectories(axes[1])
    fig.suptitle("The order parameter $\\psi$ and the composition simplex", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(HERE, "order_parameter_geometry.png"), dpi=140, bbox_inches="tight")
    print("Saved order_parameter_geometry.png")


# --------------------------------------------------------------------------- #
# Network evolution: a small graph, real Glauber dynamics, watched by eye.
# --------------------------------------------------------------------------- #
def run_snapshots(G, eps, T, sweeps, snap_at, seed):
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    N = len(nodes)
    neighbors = [list(G.neighbors(n)) for n in nodes]
    P = [[1.0, -eps, eps], [eps, 1.0, -eps], [-eps, eps, 1.0]]
    states = rng.integers(0, 3, size=N).tolist()
    snaps = {}
    if 0 in snap_at:
        snaps[0] = list(states)
    for t in range(1, sweeps + 1):
        order = rng.integers(0, N, size=N)
        proposals = rng.integers(1, 3, size=N)
        accepts = rng.random(size=N)
        for i in range(N):
            n = order[i]
            cur = states[n]
            new = (cur + proposals[i]) % 3
            nbrs = neighbors[n]
            if not nbrs:
                continue
            Pn, Pc = P[new], P[cur]
            dU = sum(Pn[states[m]] - Pc[states[m]] for m in nbrs)
            if accepts[i] * (1.0 + np.exp(-dU / T)) < 1.0:
                states[n] = new
        if t in snap_at:
            snaps[t] = list(states)
    return snaps


def make_network_evolution():
    N, K, GSEED, EPS, T = 60, 6, 3, 0.30, 0.65
    G = build_graph("ER", N, K, seed=GSEED)
    snap_at = [0, 6, 30]
    snaps = run_snapshots(G, EPS, T, sweeps=30, snap_at=snap_at, seed=11)
    layout = nx.spring_layout(G, seed=42, k=1.4 / np.sqrt(N))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    labels = ["$t=0$ (random start)", "$t=6$ (majority forming)", "$t=30$ (consensus)"]
    rows = {"node": list(G.nodes())}
    for ax, t, label in zip(axes, snap_at, labels):
        colors = [SCOLORS[s] for s in snaps[t]]
        nx.draw_networkx_edges(G, layout, ax=ax, edge_color="lightgray", width=0.5)
        nx.draw_networkx_nodes(G, layout, ax=ax, node_color=colors, node_size=90,
                              edgecolors="black", linewidths=0.4)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
        rows[f"strategy_t{t}"] = snaps[t]
    handles = [mpatches.Patch(color=c, label=n) for c, n in zip(SCOLORS, SNAMES)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
             bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.suptitle(f"How the network orders: one ER graph ($N{{=}}{N}$, "
                f"$\\langle k\\rangle{{=}}{K}$, $\\varepsilon{{=}}{EPS:g}$), "
                "same node layout throughout", fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    fig.savefig(os.path.join(HERE, "network_evolution.png"), dpi=140, bbox_inches="tight")
    print("Saved network_evolution.png")
    save_table(os.path.join(HERE, "network_evolution.csv"), rows)


def main():
    make_order_parameter_geometry()
    make_network_evolution()


if __name__ == "__main__":
    main()
