"""
The zealot story as a time signal: who led at the start, what the zealots did,
who won in the end.

Four scenarios on the same ER graph (N=800, <k>=10, seed 3), recorded per
sweep via the engine's --timeseries flag:
  * eps=0.3, z=0      -- clean ordering: whoever leads early snowballs.
  * eps=0.3, z=5%     -- Rock-zealots backfire: Paper (Rock's predator) wins.
  * eps=0.3, z=20%    -- a large faction: pinning vs frustration.
  * eps=0.9, z=10%    -- cycling phase: the endless chase, zealots or not.
Besides the figure, a machine-readable story table (timeseries_story.csv)
records the initial composition/leader, when the final winner took the lead,
when it crossed 50%, and the final state -- the report narrates from it.
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.graphs import build_graph, write_edgelist
from common.io import save_table
from common import runner

HERE = os.path.dirname(os.path.abspath(__file__))
N, K, SEED, SWEEPS = 800, 10, 3, 1500
SCEN = [("clean ordering", 0.3, 0.0), ("backfire", 0.3, 0.05),
        ("large faction", 0.3, 0.20), ("cycling", 0.9, 0.10)]
NAMES = ["Rock", "Paper", "Scissors"]
COLS = ["tab:red", "tab:blue", "tab:green"]


def main():
    runner.ensure_engine()
    el = write_edgelist(build_graph("ER", N, K, seed=SEED),
                        os.path.join(HERE, "_tsg.edgelist"))
    curves = {}
    story = {"epsilon": [], "z": [], "r0": [], "p0": [], "s0": [],
             "initial_leader": [], "winner": [], "t_lead": [], "t_majority": [],
             "r_final": [], "p_final": [], "s_final": []}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, (label, eps, z) in zip(axes.flat, SCEN):
        tsp = os.path.join(HERE, "_ts_tmp.csv")
        runner.run_engine(el, eps, zealot_frac=z, zealot_strategy=0, seed=SEED,
                          sweeps=SWEEPS, timeseries=tsp)
        d = np.genfromtxt(tsp, delimiter=",", names=True)
        os.remove(tsp)
        t = d["t"]
        fr = np.stack([d["r"], d["p"], d["s"]], axis=1)
        tag = f"e{int(eps*100)}_z{int(z*100)}"
        if "t" not in curves:
            curves["t"] = t
        for j, key in enumerate("rps"):
            curves[f"{key}_{tag}"] = fr[:, j]

        final = fr[-500:].mean(axis=0)                 # time-average of the tail
        winner = int(np.argmax(final))
        lead = np.argmax(fr, axis=1)
        not_lead = np.where(lead != winner)[0]
        t_lead = int(t[not_lead[-1] + 1]) if len(not_lead) and not_lead[-1] + 1 < len(t) \
            else int(t[0])
        below = np.where(fr[:, winner] <= 0.5)[0]
        t_maj = int(t[below[-1] + 1]) if len(below) and below[-1] + 1 < len(t) else -1
        story["epsilon"].append(eps); story["z"].append(z)
        for key, v in zip(("r0", "p0", "s0"), fr[0]):
            story[key].append(v)
        story["initial_leader"].append(int(np.argmax(fr[0])))
        story["winner"].append(winner)
        story["t_lead"].append(t_lead); story["t_majority"].append(t_maj)
        for key, v in zip(("r_final", "p_final", "s_final"), final):
            story[key].append(v)

        # consensus on a dense random graph forms within ~10 sweeps, so a log
        # time axis is the only way to see the early race AND the steady state
        tp = t + 1                                     # log axis: shift t=0 to 1
        for j in range(3):
            ax.plot(tp, fr[:, j], color=COLS[j], lw=1.1, label=NAMES[j])
        ax.set_xscale("log")
        ax.axhline(0.5, color="gray", ls="--", lw=0.7)
        ax.axhline(1/3, color="gray", ls=":", lw=0.7)
        if z > 0:
            ax.axhline(z, color=COLS[0], ls="-.", lw=0.8, alpha=0.7)
            ax.annotate(f"zealot floor z={z:g}", (tp[-1], z), fontsize=7,
                        color=COLS[0], ha="right", va="bottom")
        if t_maj > 0:
            ax.axvline(t_maj + 1, color=COLS[winner], ls=":", lw=1)
            ax.annotate(f"{NAMES[winner]} majority\n(sweep {t_maj})",
                        (t_maj + 1.5, 0.55), fontsize=8, color=COLS[winner])
        ax.annotate(f"start ($t{{=}}0$): {NAMES[int(np.argmax(fr[0]))]} leads "
                    f"({fr[0].max():.2f})", (0.02, 0.93), xycoords="axes fraction",
                    fontsize=8)
        ax.set_title(f"{label}: $\\varepsilon={eps}$, $z={z:g}$ Rock-zealots"
                     + (f"  $\\to$ {NAMES[winner]} wins ({final[winner]:.2f})"
                        if eps < 0.5 else "  $\\to$ no winner (cycling)"))
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8, loc="center right")
    for ax in axes[1]:
        ax.set_xlabel("sweep $t+1$ (log scale)")
    for ax in axes[:, 0]:
        ax.set_ylabel("population fraction")
    fig.suptitle("Zealot experiments as time signals: per-sweep $(r,p,s)$ on one "
                 f"ER graph ($N{{=}}{N}$, $\\langle k\\rangle{{=}}{K}$, seed {SEED})",
                 fontweight="bold")
    fig.tight_layout()
    os.remove(el)
    fig.savefig(os.path.join(HERE, "timeseries.png"), dpi=130)
    print("Saved timeseries.png")
    save_table(os.path.join(HERE, "timeseries.csv"), curves)
    save_table(os.path.join(HERE, "timeseries_story.csv"), story)
    for i, (label, eps, z) in enumerate(SCEN):
        w_ = story["winner"][i]
        print(f"{label} (eps={eps}, z={z}): start "
              f"r={story['r0'][i]:.2f} p={story['p0'][i]:.2f} s={story['s0'][i]:.2f} "
              f"(leader {NAMES[story['initial_leader'][i]]}), winner {NAMES[w_]}, "
              f"lead from sweep {story['t_lead'][i]}, majority at "
              f"{story['t_majority'][i]}, final "
              f"r={story['r_final'][i]:.2f} p={story['p_final'][i]:.2f} "
              f"s={story['s_final'][i]:.2f}")


if __name__ == "__main__":
    main()
