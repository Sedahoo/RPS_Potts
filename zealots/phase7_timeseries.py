"""
Phase 7 as a time signal: the z-sweep of experiment.py (zealots.png) only shows
the time-AVERAGED conversion and m_psi per z. Here every z on that same grid is
re-run with --timeseries, on one fixed ER graph, so we can watch HOW the
non-monotonic conversion curve (dip near z~0.05, partial recovery by z~0.20 in
the ordering phase) actually plays out sweep by sweep -- and how the cycling
phase's flat, weak response looks in time.

Two signals per phase, one curve per z (colour-graded, light=low z, dark=high z):
  * conversion(t) = fraction of FREE nodes playing Rock -- (r(t)-z)/(1-z),
    derived from the engine's per-sweep global r(t) (zealots included).
  * |psi(t)|      = instantaneous order-parameter magnitude (NOT time-averaged;
    the report's m_psi averages the complex psi over a burn-in window first --
    this is the un-averaged magnitude at each single sweep, a genuine
    high-frequency diagnostic of how "settled" the state is moment to moment).
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.graphs import build_graph, write_edgelist
from common.observables import psi_series
from common.io import save_table
from common import runner

HERE = os.path.dirname(os.path.abspath(__file__))
N, K, SEED, SWEEPS = 800, 10, 1, 1500
Z_VALS = np.linspace(0.0, 0.20, 17)           # identical grid to experiment.py
REGIMES = [("Ordering phase", 0.3, "order"), ("Cycling phase", 0.9, "cycle")]
CMAP = plt.cm.viridis


def run_one(el, eps, z, tag):
    tsp = os.path.join(HERE, f"_p7ts_{tag}_{int(round(z * 1000)):03d}.csv")
    runner.run_engine(el, eps, zealot_frac=float(z), zealot_strategy=0, seed=SEED,
                      sweeps=SWEEPS, timeseries=tsp)
    d = np.genfromtxt(tsp, delimiter=",", names=True)
    os.remove(tsp)
    return d["t"], d["r"], d["p"], d["s"]


def main():
    runner.ensure_engine()
    el = write_edgelist(build_graph("ER", N, K, seed=SEED),
                        os.path.join(HERE, "_p7ts.edgelist"))
    norm = Normalize(vmin=Z_VALS.min(), vmax=Z_VALS.max())
    curves = {}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex="col")

    for col, (label, eps, tag) in enumerate(REGIMES):
        jobs = [(el, eps, z, tag) for z in Z_VALS]
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
            results = list(ex.map(lambda j: run_one(*j), jobs))

        ax_conv, ax_psi = axes[0, col], axes[1, col]
        for z, (t, r, p, s) in zip(Z_VALS, results):
            conv = (r - z) / (1.0 - z)
            psi_mag = np.abs(psi_series(np.stack([r, p, s], axis=1)))
            color = CMAP(norm(z))
            tp = t + 1                          # log axis: shift t=0 to 1
            ax_conv.plot(tp, conv, color=color, lw=1.0)
            ax_psi.plot(tp, psi_mag, color=color, lw=1.0)
            zt = f"{z:.3f}".replace(".", "p")
            curves[f"t_{tag}"] = t
            curves[f"conversion_{tag}_z{zt}"] = conv
            curves[f"psi_mag_{tag}_z{zt}"] = psi_mag

        for ax in (ax_conv, ax_psi):
            ax.set_xscale("log"); ax.set_xlim(1, SWEEPS)
        ax_conv.axhline(1/3, color="gray", ls="--", lw=0.8)
        ax_conv.set_ylim(-0.02, 1.02)
        ax_psi.set_ylim(-0.02, 1.02)
        ax_conv.set_title(f"{label} ($\\varepsilon={eps}$)")

    axes[0, 0].set_ylabel("conversion$(t)$\n(free nodes playing Rock)")
    axes[1, 0].set_ylabel(r"$|\psi(t)|$  (instantaneous)")
    for ax in axes[1]:
        ax.set_xlabel("sweep $t+1$ (log scale)")

    sm = ScalarMappable(norm=norm, cmap=CMAP); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.025, pad=0.02)
    cbar.set_label("zealot fraction $z$ (Rock-zealots)")

    fig.suptitle("Phase 7 as a time signal: per-sweep conversion and $|\\psi(t)|$ "
                 f"across the $z$-sweep (one ER graph, $N{{=}}{N}$, "
                 f"$\\langle k\\rangle{{=}}{K}$, seed {SEED})", fontweight="bold")
    os.remove(el)
    fig.savefig(os.path.join(HERE, "phase7_timeseries.png"), dpi=130,
               bbox_inches="tight")
    print("Saved phase7_timeseries.png")
    save_table(os.path.join(HERE, "phase7_timeseries.csv"), curves)


if __name__ == "__main__":
    main()
