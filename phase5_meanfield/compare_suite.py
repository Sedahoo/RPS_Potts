"""
Phase 5: the MC vs HMF vs DMF comparison suite -- the analytical heart of
Project B.

For a chosen graph type and average degree, we sweep epsilon and compute m_psi
three ways:
  * MC   -- the stochastic ground truth (C++ engine on the real graph).
  * HMF  -- homogeneous mean field (scalar <k> only).
  * DMF  -- degree-based mean field (uses the full P(k)).
We overlay the curves and report RMSE(model, MC). The thesis question:
does DMF track the MC truth better than HMF, especially on heterogeneous BA?

Run:  ../.venv/bin/python compare_suite.py --graph BA --k 10
"""

import os, sys, argparse, subprocess
from collections import Counter
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase1_hmf"))
from hmf import run as hmf_run, order_parameter as hmf_order
import dmf

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "..", "phase3_cpp", "mc_simulation")


def build(graph_type, n, k, seed=1):
    if graph_type == "ER":
        G = nx.erdos_renyi_graph(n, k / (n - 1), seed=seed)
    else:
        G = nx.barabasi_albert_graph(n, max(1, k // 2), seed=seed)
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G = nx.convert_node_labels_to_integers(G)
    edgelist = os.path.join(HERE, f"_g_{graph_type}_k{k}.edgelist")
    nx.write_edgelist(G, edgelist, data=False)
    degrees = [d for _, d in G.degree()]
    Nnow = G.number_of_nodes()
    pk = {k_: c / Nnow for k_, c in Counter(degrees).items()}
    mean_deg = sum(d for d in degrees) / Nnow
    return edgelist, pk, mean_deg


def mc_mpsi(edgelist, eps, T, sweeps=1500):
    burn = int(sweeps * 0.3)
    out = subprocess.run(
        [BIN, "--graph", edgelist, "--epsilon", str(eps), "--temp", str(T),
         "--sweeps", str(sweeps), "--burn-in", str(burn), "--seed", "1"],
        capture_output=True, text=True).stdout
    return float(out.split()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="BA", choices=["ER", "BA"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--temp", type=float, default=0.65)
    a = ap.parse_args()

    eps_vals = np.linspace(0.0, 1.0, 26)
    edgelist, pk, mean_deg = build(a.graph, a.n, a.k, seed=1)
    print(f"{a.graph}: <k>={mean_deg:.2f}, {len(pk)} distinct degrees "
          f"(max={max(pk)})")

    mc  = np.array([mc_mpsi(edgelist, e, a.temp) for e in eps_vals])
    hmf = np.array([hmf_order(hmf_run(e, k=mean_deg, T=a.temp))[0] for e in eps_vals])
    dmf_ = np.array([dmf.order_parameter(dmf.run(pk, e, T=a.temp)) for e in eps_vals])
    os.remove(edgelist)

    rmse_hmf = np.sqrt(np.mean((hmf - mc) ** 2))
    rmse_dmf = np.sqrt(np.mean((dmf_ - mc) ** 2))

    plt.figure(figsize=(8, 5.5))
    plt.plot(eps_vals, mc,   "o-", color="black",      ms=4, label="MC (ground truth)")
    plt.plot(eps_vals, hmf,  "s--", color="tab:orange", ms=4,
             label=f"HMF  (RMSE={rmse_hmf:.3f})")
    plt.plot(eps_vals, dmf_, "^--", color="tab:green",  ms=4,
             label=f"DMF  (RMSE={rmse_dmf:.3f})")
    plt.axhline(0.5, color="gray", ls=":", lw=1)
    plt.xlabel("epsilon"); plt.ylabel(r"$m_\psi$")
    plt.title(f"MC vs HMF vs DMF  ({a.graph}, N={a.n}, <k>~{a.k}, T={a.temp})")
    plt.ylim(-0.02, 1.05); plt.legend()
    plt.tight_layout()
    out = f"comparison_suite_{a.graph}_k{a.k}.png"
    plt.savefig(out, dpi=130)
    print(f"Saved {out}")
    print(f"  RMSE(HMF, MC) = {rmse_hmf:.4f}")
    print(f"  RMSE(DMF, MC) = {rmse_dmf:.4f}")
    print(f"  winner: {'DMF' if rmse_dmf < rmse_hmf else 'HMF'}")


if __name__ == "__main__":
    main()
