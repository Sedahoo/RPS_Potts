"""
S7 (report Sec. 3): sensitivity of the HMF map to its initial composition.

The production HMF runs start from (0.40, 0.35, 0.25). Here the same eps sweep
runs from four different starting compositions at k=10 and k=20. Hypothesis
(sensitivity/HYPOTHESES.md): biased inits are equivalent (same attractor);
the near-symmetric init is the exception near eps_c only, because
(1/3, 1/3, 1/3) is an unstable fixed point whose escape transient can outlast
the measurement window where the map slows down.
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.meanfield import hmf_run
from common.observables import order_parameter, eps_crossing
from common.io import save_table

HERE = os.path.dirname(os.path.abspath(__file__))
INITS = [(0.3334, 0.3333, 0.3333), (0.40, 0.35, 0.25),
         (0.50, 0.30, 0.20), (0.90, 0.05, 0.05)]
LABELS = ["near-symmetric", "default", "biased", "strongly biased"]
KS = [10, 20]
EPS = np.linspace(0, 1, 81)
STEPS = 4000


def main():
    curves = {"epsilon": EPS}
    summary = {"k": [], "init_r": [], "init_p": [], "init_s": [], "eps_c": []}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, k in zip(axes, KS):
        for (init, lab) in zip(INITS, LABELS):
            m = np.array([order_parameter(hmf_run(e, k=k, steps=STEPS, init=init))
                          for e in EPS])
            curves[f"m_k{k}_r{int(init[0]*1e4)}"] = m
            ec = eps_crossing(EPS, m)
            summary["k"].append(k)
            for name, v in zip(("init_r", "init_p", "init_s"), init):
                summary[name].append(v)
            summary["eps_c"].append(ec)
            ax.plot(EPS, m, lw=1.6, label=f"{lab} {init}  ($\\varepsilon_c$={ec:.3f})")
        ax.axhline(0.5, color="gray", ls="--", lw=0.8)
        ax.set_title(f"HMF, $k={k}$"); ax.set_xlabel(r"$\varepsilon$")
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel(r"$m_\psi$")
    fig.suptitle("S7: the HMF transition is subcritical --- a bistable window "
                 "where the initial composition picks the attractor",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "sens_mf_init.png"), dpi=130)
    print("Saved sens_mf_init.png")
    save_table(os.path.join(HERE, "sens_mf_init.csv"), summary)
    save_table(os.path.join(HERE, "sens_mf_init_curves.csv"), curves)

    for k in KS:
        ecs = [summary["eps_c"][i] for i in range(len(summary["k"]))
               if summary["k"][i] == k]
        # ecs[1:] are the three biased inits: their eps_c range IS the bistable
        # window of the map (both attractors stable; the start decides)
        print(f"k={k}: eps_c spread across inits = {max(ecs)-min(ecs):.4f}; "
              f"bistable window from biased inits = [{min(ecs[1:]):.3f}, "
              f"{max(ecs[1:]):.3f}] (width {max(ecs[1:])-min(ecs[1:]):.4f})")


if __name__ == "__main__":
    main()
