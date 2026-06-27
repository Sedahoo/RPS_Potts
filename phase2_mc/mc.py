"""
Phase 2: Agent-level Monte Carlo of Potts-RPS dynamics ON A NETWORK.

The conceptual jump from Phase 1:
  Phase 1 (HMF) tracked only POPULATION FRACTIONS and assumed everyone feels
  the average mix. There were no individuals.
  Here, every node is a real AGENT holding one strategy, and it only feels its
  OWN neighbours in the graph. The dynamics are now STOCHASTIC -- run it twice
  and you get different trajectories. This is the "ground truth" the whole
  project's mean-field models try to approximate.

This is the Python version of the original repo's `mc_simulation.cpp`. It is
slow but easy to read and debug; in Phase 3 we port the hot loop to C++.

The Monte Carlo step (Glauber / Metropolis-like):
  1. Pick a random node n.
  2. Propose a different strategy s' (one of the other two, picked at random).
  3. Compute the payoff change if n switched, summed over its neighbours:
        dU = sum_neighbours ( P[s'][neighbour] - P[s][neighbour] )
  4. Accept the switch with probability  logistic(dU / T) = 1/(1+exp(-dU/T)).
     Higher payoff -> more likely to accept; temperature T allows uphill moves.
  One "sweep" = N such attempts (so on average every node is tried once).
"""

import argparse
import numpy as np
import networkx as nx


def build_graph(graph_type, n, avg_degree, seed):
    """Build an ER or BA graph and return neighbour lists (its largest
    connected component, so every node has someone to interact with)."""
    if graph_type == "ER":
        p = avg_degree / (n - 1)
        G = nx.erdos_renyi_graph(n, p, seed=seed)
    elif graph_type == "BA":
        m = max(1, avg_degree // 2)          # BA adds m edges/node -> <k> ~ 2m
        G = nx.barabasi_albert_graph(n, m, seed=seed)
    else:
        raise ValueError("graph_type must be 'ER' or 'BA'")

    if not nx.is_connected(G):               # keep the giant component
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G = nx.convert_node_labels_to_integers(G)
    # neighbour lists as plain Python lists -> fast to iterate in the hot loop
    return [list(G.neighbors(i)) for i in range(G.number_of_nodes())]


def order_parameter(fracs, burn_in_frac=0.3):
    """Same complex RPS order parameter as Phase 1, but measured from the
    stochastic fraction time series. m_psi ~ 1 ordered, ~ 0 cycling."""
    fracs = np.asarray(fracs)
    r, p, s = fracs[:, 0], fracs[:, 1], fracs[:, 2]
    psi = r + p * np.exp(1j * 2 * np.pi / 3) + s * np.exp(1j * 4 * np.pi / 3)
    start = int(len(fracs) * burn_in_frac)
    return np.abs(np.mean(psi[start:]))


def run_mc(neighbors, eps, T=0.65, sweeps=1500, seed=0):
    """Run the Monte Carlo and return the per-sweep (r, p, s) fractions."""
    rng = np.random.default_rng(seed)
    N = len(neighbors)
    # payoff matrix P = I + eps*skew
    P = [[1.0, -eps, eps], [eps, 1.0, -eps], [-eps, eps, 1.0]]

    states = rng.integers(0, 3, size=N).tolist()      # random initial strategies
    fracs = np.empty((sweeps, 3))

    # pre-draw the random numbers we need each sweep (fast + reproducible)
    for t in range(sweeps):
        nodes = rng.integers(0, N, size=N)            # which node to try
        proposals = rng.integers(1, 3, size=N)        # +1 or +2 (mod 3) -> other strategy
        accepts = rng.random(size=N)                  # acceptance dice
        for i in range(N):
            n = nodes[i]
            cur = states[n]
            new = (cur + proposals[i]) % 3
            nbrs = neighbors[n]
            if not nbrs:
                continue
            Pn, Pc = P[new], P[cur]
            dU = 0.0
            for m in nbrs:
                sm = states[m]
                dU += Pn[sm] - Pc[sm]
            # accept with probability logistic(dU/T)
            if accepts[i] * (1.0 + np.exp(-dU / T)) < 1.0:
                states[n] = new

        c0 = states.count(0); c1 = states.count(1)
        fracs[t] = (c0 / N, c1 / N, (N - c0 - c1) / N)
    return fracs


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
    neighbors = build_graph(args.graph, args.n, args.k, args.seed)
    Nactual = len(neighbors)
    print(f"{args.graph} graph: N={Nactual}, <k>~{args.k}, T={args.temp}")

    cases = [("Low epsilon (ordering)", 0.2), ("High epsilon (cycling)", 0.9)]
    fig, axes = plt.subplots(2, len(cases), figsize=(11, 7), sharex=True)
    for col, (title, eps) in enumerate(cases):
        fracs = run_mc(neighbors, eps, T=args.temp, sweeps=args.sweeps, seed=args.seed)
        m_psi = order_parameter(fracs)

        ax = axes[0, col]
        ax.plot(fracs[:, 0], color="tab:red", lw=0.8, label="rock")
        ax.plot(fracs[:, 1], color="tab:green", lw=0.8, label="paper")
        ax.plot(fracs[:, 2], color="tab:blue", lw=0.8, label="scissors")
        ax.set_title(f"{title}  (eps={eps})")
        ax.set_ylabel("fraction"); ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="upper right", fontsize=8)

        psi = (fracs[:, 0] + fracs[:, 1] * np.exp(1j * 2 * np.pi / 3)
               + fracs[:, 2] * np.exp(1j * 4 * np.pi / 3))
        ax = axes[1, col]
        ax.plot(np.abs(psi), color="black", lw=0.8)
        ax.set_title(f"m_psi = {m_psi:.3f}  ->  "
                     f"{'ORDERED' if m_psi > 0.5 else 'DISORDERED'}")
        ax.set_ylabel("|psi(t)|"); ax.set_xlabel("sweep"); ax.set_ylim(-0.02, 1.02)
        print(f"  eps={eps}:  m_psi={m_psi:.3f}")

    fig.suptitle(f"Phase 2: Monte Carlo on {args.graph} network "
                 f"(N={Nactual}, <k>~{args.k}, T={args.temp})", fontweight="bold")
    fig.tight_layout()
    out = f"mc_phase2_{args.graph}.png"
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
