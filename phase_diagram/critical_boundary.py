"""
Extract the critical boundary eps_c(<k>) from the phase-diagram heatmaps.

The heatmaps show the ordered/cycling regions; this pulls out the boundary
itself as a curve, for ER and BA, and overlays the HMF prediction computed
at EVERY degree the MC was run on (k=2..80), on the identical epsilon grid
and with the identical m_psi=0.5 crossing estimator, so all three curves are
method-for-method comparable. The plotted HMF curve uses the standard
(0.40, 0.35, 0.25) init; because the mean-field transition is subcritical
(see dynamics/ and sensitivity/sens_mf_init.py), a second HMF sweep from a
strongly-ordered init (0.98, 0.01, 0.01) is also computed -- not plotted, but
saved to the CSV and used for the write-up -- to check whether the standard
-init curve dipping below the MC at high k is a real mean-field failure or an
init-convention artefact of the bistable window.

Run AFTER run.sh (needs phase_diagram_{ER,BA}.csv).
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.io import save_table
from common.meanfield import hmf_run
from common.observables import order_parameter

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


def mc_eps_grid(graph="ER"):
    d = np.genfromtxt(os.path.join(HERE, f"phase_diagram_{graph}.csv"),
                      delimiter=",", names=True)
    return np.array(sorted(set(d["epsilon"])))


def hmf_boundary(ks, eps_values, T=0.65, steps=4000, init=(0.40, 0.35, 0.25)):
    """HMF eps_c at every MC degree: same eps grid, same estimator.

    The HMF map (common.meanfield.hmf_step) is iterated for each (k, eps)
    tile of the MC grid, m_psi taken exactly as for the MC data, and the
    boundary read off with the same interpolated crossing.
    """
    ec = []
    for k in ks:
        m = np.array([order_parameter(hmf_run(e, k=float(k), T=T, steps=steps,
                                               init=init))
                      for e in eps_values])
        ec.append(eps_c(eps_values, m))
    return np.array(ec)


def main():
    ks_er, ec_er = mc_boundary("ER")
    ks_ba, ec_ba = mc_boundary("BA")
    eps_values = mc_eps_grid("ER")
    print(f"HMF boundary on the full MC grid: {len(ks_er)} degrees x "
          f"{len(eps_values)} epsilons ...")
    ec_h = hmf_boundary(ks_er, eps_values)                       # standard init
    ec_h_ord = hmf_boundary(ks_er, eps_values, init=(0.98, 0.01, 0.01))  # ordered
    # the mean-field transition is subcritical: ec_h is the window's lower
    # edge, ec_h_ord its upper edge (not plotted -- used only for the window
    # check below, per the note in the module docstring)
    ec_h_lo, ec_h_hi = np.minimum(ec_h, ec_h_ord), np.maximum(ec_h, ec_h_ord)
    inside = (ec_er >= ec_h_lo) & (ec_er <= ec_h_hi)

    plt.figure(figsize=(8, 5.5))
    plt.fill_between(ks_er, 0, ec_er, color="tab:red", alpha=0.08)
    plt.fill_between(ks_er, ec_er, 1, color="tab:blue", alpha=0.08)
    plt.fill_between(ks_er, np.minimum(ec_er, ec_h), np.maximum(ec_er, ec_h),
                     color="tab:green", alpha=0.15, label="HMF$-$MC gap")
    plt.plot(ks_er, ec_er, "o-", color="tab:blue", label="MC boundary (ER)")
    plt.plot(ks_ba, ec_ba, "s-", color="tab:orange", ms=5, label="MC boundary (BA)")
    plt.plot(ks_er, ec_h, "D--", color="tab:green", ms=5,
             label="HMF prediction (standard init, same grid & estimator)")
    plt.text(28, 0.30, "ORDERED\n(consensus)", ha="center", color="darkred", fontsize=11)
    plt.text(10, 0.88, "CYCLING", ha="center", color="darkblue", fontsize=11)
    plt.xlabel(r"average degree $\langle k\rangle$")
    plt.ylabel(r"$\varepsilon_c$  (interpolated $m_\psi=0.5$ crossing)")
    plt.title("Critical boundary $\\varepsilon_c(\\langle k\\rangle)$: "
              "MC (ER vs BA) and the HMF prediction at every degree")
    plt.xticks(ks_er[::2]); plt.grid(alpha=0.25, lw=0.5)
    plt.ylim(0, 1.0); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(os.path.join(HERE, "critical_boundary.png"), dpi=130)
    print("Saved critical_boundary.png")

    save_table(os.path.join(HERE, "critical_boundary.csv"),
               {"k": ks_er, "eps_c_ER": ec_er, "eps_c_BA": ec_ba,
                "eps_c_HMF": ec_h, "eps_c_HMF_ordered_init": ec_h_ord})
    gap = np.max(np.abs(ec_er - ec_ba))
    print(f"max |eps_c(ER) - eps_c(BA)| = {gap:.3f}  "
          f"(ER and BA boundaries {'coincide' if gap < 0.1 else 'differ'})")
    dh = ec_h - ec_er
    print(f"HMF(standard) - MC(ER): max +{dh.max():.3f} at k={ks_er[dh.argmax()]}, "
          f"min {dh.min():+.3f} at k={ks_er[dh.argmin()]}")
    print(f"MC(ER) inside the HMF bistable window [standard,ordered]: "
          f"{inside.sum()}/{len(ks_er)} degrees, "
          f"first at k={ks_er[inside.argmax()] if inside.any() else 'never'}")


if __name__ == "__main__":
    main()
