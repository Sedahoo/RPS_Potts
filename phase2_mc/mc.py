"""
Phase 2: Agent-level Monte Carlo ON A NETWORK (pure-Python reference).

Unlike HMF, every node is a real agent that feels only its own neighbours, and
the dynamics are stochastic. This is the readable reference; the production
version is the C++ engine in drivers/. Engine + helpers live in common/.
"""

import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.graphs import build_graph, neighbor_lists
from common.mc_python import run_mc
from common.observables import order_parameter, psi_series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="ER", choices=["ER", "BA"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--temp", type=float, default=0.65)
    ap.add_argument("--sweeps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    G = build_graph(args.graph, args.n, args.k, seed=args.seed)
    neighbors = neighbor_lists(G)
    Nactual = G.number_of_nodes()
    print(f"{args.graph} graph: N={Nactual}, <k>~{args.k}, T={args.temp}")

    cases = [("Low epsilon (ordering)", 0.2), ("High epsilon (cycling)", 0.9)]
    fig, axes = plt.subplots(2, len(cases), figsize=(11, 7), sharex=True)
    for col, (title, eps) in enumerate(cases):
        fracs = run_mc(neighbors, eps, T=args.temp, sweeps=args.sweeps, seed=args.seed)
        m_psi = order_parameter(fracs, burn_in_frac=0.3)

        ax = axes[0, col]
        ax.plot(fracs[:, 0], color="tab:red", lw=0.8, label="rock")
        ax.plot(fracs[:, 1], color="tab:green", lw=0.8, label="paper")
        ax.plot(fracs[:, 2], color="tab:blue", lw=0.8, label="scissors")
        ax.set_title(f"{title}  (eps={eps})")
        ax.set_ylabel("fraction"); ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="upper right", fontsize=8)

        ax = axes[1, col]
        ax.plot(np.abs(psi_series(fracs)), color="black", lw=0.8)
        ax.set_title(f"m_psi = {m_psi:.3f}  ->  "
                     f"{'ORDERED' if m_psi > 0.5 else 'DISORDERED'}")
        ax.set_ylabel("|psi(t)|"); ax.set_xlabel("sweep"); ax.set_ylim(-0.02, 1.02)
        print(f"  eps={eps}:  m_psi={m_psi:.3f}")

    fig.suptitle(f"Phase 2: Monte Carlo on {args.graph} network "
                 f"(N={Nactual}, <k>~{args.k}, T={args.temp})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"mc_phase2_{args.graph}.png", dpi=130)
    print(f"Saved mc_phase2_{args.graph}.png")


if __name__ == "__main__":
    main()
