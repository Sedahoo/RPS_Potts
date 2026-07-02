"""
The collapse test: does a DAMAGED network behave like a PRISTINE network of the
same average degree?

From defects.csv, each (defect type, f) gives a point (resulting <k>, eps_c).
We plot those points over the pristine-ER critical boundary eps_c(<k>) extracted
from phase_diagram_ER.csv. If damaged and undamaged networks fall on one curve,
order-stability depends on the effective <k> only -- the strongest version of
the defects finding.

Run AFTER experiment_defects.py and phase_diagram/run.sh ER.
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.io import save_table

HERE = os.path.dirname(os.path.abspath(__file__))


def eps_c(eps, m, thr=0.5):
    o = np.argsort(eps)
    eps, m = np.asarray(eps)[o], np.asarray(m)[o]
    below = np.where(m < thr)[0]
    if len(below) == 0:
        return eps[-1]
    i = below[0]
    if i == 0:
        return eps[0]
    e0, e1, m0, m1 = eps[i - 1], eps[i], m[i - 1], m[i]
    return e0 + (thr - m0) * (e1 - e0) / (m1 - m0)


def main():
    d = np.genfromtxt(os.path.join(HERE, "defects.csv"), delimiter=",", names=True)
    pts = {0: [], 1: []}                       # type -> [(mean_k, eps_c, f)]
    for t in (0, 1):
        for f in sorted(set(d["f"][d["defect_type_0edge_1node"] == t])):
            sub = d[(d["defect_type_0edge_1node"] == t) & (d["f"] == f)]
            pts[t].append((sub["mean_k"][0], eps_c(sub["epsilon"], sub["m_psi"]), f))

    pd = np.genfromtxt(os.path.join(HERE, "..", "phase_diagram", "phase_diagram_ER.csv"),
                       delimiter=",", names=True)
    ks = np.array(sorted(set(pd["degree"].astype(int))))
    ec_prist = np.array([eps_c(pd["epsilon"][pd["degree"] == k],
                               pd["m_psi"][pd["degree"] == k]) for k in ks])

    plt.figure(figsize=(8, 5.5))
    plt.plot(ks, ec_prist, "-", color="gray", lw=2, alpha=0.7,
             label="pristine ER boundary (phase diagram, N=800)")
    ek, ee, ef = zip(*pts[0]); nk, ne, nf = zip(*pts[1])
    plt.plot(ek, ee, "o", color="tab:red", ms=9, label="edge defects (N=1000, damaged)")
    plt.plot(nk, ne, "s", color="tab:blue", ms=7, label="node defects (N=1000, damaged)")
    for k, e, f in pts[0]:
        plt.annotate(f"f={f:.1f}", (k, e), textcoords="offset points",
                     xytext=(6, 6), fontsize=8, color="tab:red")
    plt.xlabel(r"resulting average degree $\langle k\rangle$")
    plt.ylabel(r"$\varepsilon_c$")
    plt.title("Collapse test: damaged networks land on the pristine "
              "$\\varepsilon_c(\\langle k\\rangle)$ curve")
    plt.xlim(0, 42); plt.ylim(0, 0.9); plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "defects_collapse.png"), dpi=130)
    print("Saved defects_collapse.png")

    save_table(os.path.join(HERE, "defects_collapse.csv"),
               {"f": np.array(ef), "mean_k_edge": np.array(ek),
                "eps_c_edge": np.array(ee), "mean_k_node": np.array(nk),
                "eps_c_node": np.array(ne)})
    dev = np.max(np.abs(np.array(ee) - np.array(ne)))
    print(f"max |eps_c(edge) - eps_c(node)| at matched f = {dev:.3f}")


if __name__ == "__main__":
    main()
