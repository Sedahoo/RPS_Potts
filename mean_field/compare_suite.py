"""
Phase 5: the MC vs HMF vs DMF comparison suite -- the analytical heart of the
project. Sweep epsilon and compute m_psi three ways (stochastic MC ground truth,
homogeneous mean field, degree-based mean field), then report RMSE(model, MC).
DMF should track MC better than HMF, especially on heterogeneous BA graphs.

Run:  ../.venv/bin/python compare_suite.py --graph BA --k 10
"""

import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.graphs import build_graph, write_edgelist, degree_dist
from common.meanfield import hmf_run, dmf_run
from common.observables import order_parameter
from common import runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="BA", choices=["ER", "BA"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--temp", type=float, default=0.65)
    a = ap.parse_args()
    runner.ensure_engine()

    eps_vals = np.linspace(0.0, 1.0, 26)
    G = build_graph(a.graph, a.n, a.k, seed=1)
    pk, mean_deg = degree_dist(G)
    edgelist = write_edgelist(G, os.path.join(os.path.dirname(__file__),
                                              f"_g_{a.graph}_k{a.k}.edgelist"))
    print(f"{a.graph}: <k>={mean_deg:.2f}, {len(pk)} distinct degrees (max={max(pk)})")

    mc  = np.array([runner.run_engine(edgelist, e, temp=a.temp)["m_psi"] for e in eps_vals])
    hmf = np.array([order_parameter(hmf_run(e, k=mean_deg, T=a.temp)) for e in eps_vals])
    dmf = np.array([order_parameter(dmf_run(pk, e, T=a.temp)) for e in eps_vals])
    os.remove(edgelist)

    rmse_hmf = np.sqrt(np.mean((hmf - mc) ** 2))
    rmse_dmf = np.sqrt(np.mean((dmf - mc) ** 2))

    plt.figure(figsize=(8, 5.5))
    plt.plot(eps_vals, mc,  "o-",  color="black",      ms=4, label="MC (ground truth)")
    plt.plot(eps_vals, hmf, "s--", color="tab:orange", ms=4, label=f"HMF  (RMSE={rmse_hmf:.3f})")
    plt.plot(eps_vals, dmf, "^--", color="tab:green",  ms=4, label=f"DMF  (RMSE={rmse_dmf:.3f})")
    plt.axhline(0.5, color="gray", ls=":", lw=1)
    plt.xlabel("epsilon"); plt.ylabel(r"$m_\psi$")
    plt.title(f"MC vs HMF vs DMF  ({a.graph}, N={a.n}, <k>~{a.k}, T={a.temp})")
    plt.ylim(-0.02, 1.05); plt.legend(); plt.tight_layout()
    out = f"comparison_suite_{a.graph}_k{a.k}.png"
    plt.savefig(out, dpi=130)
    print(f"Saved {out}")
    print(f"  RMSE(HMF, MC) = {rmse_hmf:.4f} | RMSE(DMF, MC) = {rmse_dmf:.4f} | "
          f"winner: {'DMF' if rmse_dmf < rmse_hmf else 'HMF'}")


if __name__ == "__main__":
    main()
