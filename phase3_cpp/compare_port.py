"""
Phase 3 validation: prove the C++ engine (drivers/mc_engine) matches the
pure-Python MC, and measure the speedup. Same graph through both; the RNGs
differ so m_psi agrees within stochastic noise, not to the digit.

The engine used to live here; after the refactor it lives in drivers/ and is
shared by every phase. This script just validates it.
"""

import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.graphs import build_graph, neighbor_lists, write_edgelist
from common.mc_python import run_mc
from common.observables import order_parameter
from common import runner

N, K, T, SWEEPS = 500, 10, 0.65, 1500


def main():
    runner.ensure_engine()
    G = build_graph("ER", N, K, seed=42)
    neighbors = neighbor_lists(G)
    edgelist = write_edgelist(G, os.path.join(os.path.dirname(__file__), "_g.edgelist"))

    print(f"{'eps':>5} | {'Python m_psi':>12} {'C++ m_psi':>10} | {'verdict':>11}")
    print("-" * 50)
    t_py = t_cpp = 0.0
    for eps in (0.2, 0.5, 0.7, 0.9):
        t0 = time.time()
        py = order_parameter(run_mc(neighbors, eps, T=T, sweeps=SWEEPS, seed=7),
                             burn_in_frac=0.3)
        t_py += time.time() - t0
        t0 = time.time()
        cpp = runner.run_engine(edgelist, eps, temp=T, sweeps=SWEEPS, seed=7)["m_psi"]
        t_cpp += time.time() - t0
        verdict = "agree" if (py > 0.5) == (cpp > 0.5) else "DIFFER"
        print(f"{eps:>5} | {py:>12.3f} {cpp:>10.3f} | {verdict:>11}")
    os.remove(edgelist)
    print("-" * 50)
    print(f"Python total: {t_py:6.2f}s | C++ total: {t_cpp:6.2f}s | "
          f"speedup: {t_py/max(t_cpp,1e-9):5.1f}x")


if __name__ == "__main__":
    main()
