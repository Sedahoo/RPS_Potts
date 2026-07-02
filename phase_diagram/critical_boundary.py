"""
Extract the critical boundary eps_c(<k>) from the phase-diagram heatmaps.

The heatmaps show the ordered/cycling regions; this pulls out the boundary
itself as a curve, for ER and BA, and overlays the HMF prediction from
mean_field/hmf_sweep.csv. One figure carries three claims at once:
connectivity stabilises order (curve rises), ER ~ BA (curves coincide),
and the mean field overestimates the ordered phase (HMF sits above MC).

Run AFTER run.sh (needs phase_diagram_{ER,BA}.csv) and mean_field/sweep.py.
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.io import save_table

HERE = os.path.dirname(os.path.abspath(__file__))


def eps_c(eps, m, thr=0.5):
    """Interpolated epsilon where m_psi first crosses below thr."""
    o = np.argsort(eps)
    eps, m = np.asarray(eps)[o], np.asarray(m)[o]
    below = np.where(m < thr)[0]
    if len(below) == 0:
        return eps[-1]
    i = below[0]
    if i == 0:
        return eps[0]
    # linear interpolation between the bracketing grid points
    e0, e1, m0, m1 = eps[i - 1], eps[i], m[i - 1], m[i]
    return e0 + (thr - m0) * (e1 - e0) / (m1 - m0)


def mc_boundary(graph):
    d = np.genfromtxt(os.path.join(HERE, f"phase_diagram_{graph}.csv"),
                      delimiter=",", names=True)
    ks = np.array(sorted(set(d["degree"].astype(int))))
    ec = np.array([eps_c(d["epsilon"][d["degree"] == k], d["m_psi"][d["degree"] == k])
                   for k in ks])
    return ks, ec


def hmf_boundary():
    d = np.genfromtxt(os.path.join(HERE, "..", "mean_field", "hmf_sweep.csv"),
                      delimiter=",", names=True)
    ks = [int(c[1:]) for c in d.dtype.names if c != "epsilon"]
    ec = [eps_c(d["epsilon"], d[f"k{k}"]) for k in ks]
    return np.array(ks), np.array(ec)


def main():
    ks_er, ec_er = mc_boundary("ER")
    ks_ba, ec_ba = mc_boundary("BA")
    ks_h, ec_h = hmf_boundary()
    in_range = ks_h <= ks_er.max()          # HMF sweep goes to k=200; clip to MC range

    plt.figure(figsize=(8, 5.5))
    plt.plot(ks_er, ec_er, "o-", color="tab:blue", label="MC boundary (ER)")
    plt.plot(ks_ba, ec_ba, "s-", color="tab:orange", ms=5, label="MC boundary (BA)")
    plt.plot(ks_h[in_range], ec_h[in_range], "D--", color="tab:green", ms=7,
             label="HMF prediction")
    plt.fill_between(ks_er, 0, ec_er, color="tab:red", alpha=0.08)
    plt.fill_between(ks_er, ec_er, 1, color="tab:blue", alpha=0.08)
    plt.text(28, 0.30, "ORDERED\n(consensus)", ha="center", color="darkred", fontsize=11)
    plt.text(10, 0.88, "CYCLING", ha="center", color="darkblue", fontsize=11)
    plt.xlabel(r"average degree $\langle k\rangle$")
    plt.ylabel(r"$\varepsilon_c$  (interpolated $m_\psi=0.5$ crossing)")
    plt.title("Critical boundary $\\varepsilon_c(\\langle k\\rangle)$: "
              "MC (ER vs BA) and the HMF prediction")
    plt.ylim(0, 1.0); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(os.path.join(HERE, "critical_boundary.png"), dpi=130)
    print("Saved critical_boundary.png")

    save_table(os.path.join(HERE, "critical_boundary.csv"),
               {"k": ks_er, "eps_c_ER": ec_er, "eps_c_BA": ec_ba})
    gap = np.max(np.abs(ec_er - ec_ba))
    print(f"max |eps_c(ER) - eps_c(BA)| = {gap:.3f}  "
          f"(ER and BA boundaries {'coincide' if gap < 0.1 else 'differ'})")


if __name__ == "__main__":
    main()
