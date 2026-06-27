"""
Phase 1: Homogeneous Mean-Field (HMF) model of Potts-RPS dynamics.

This is the SIMPLEST version of the whole project: no network, no agents,
no randomness. Just 3 coupled equations for the fractions of the population
playing Rock, Paper, Scissors. We pretend everyone is equally connected to
everyone else (a "well-mixed" population with average degree k).

The point of this file: watch the phase transition appear with the least
possible machinery. Low epsilon -> the population ORDERS (everyone agrees).
High epsilon -> the population CYCLES forever (R->P->S->R...).

This mirrors the original repo's `sde_simulation.cpp` (which is the *true*
homogeneous mean field, despite its filename).
"""

import numpy as np
import matplotlib.pyplot as plt


def step(r, p, s, eps, k, T):
    """Advance the (r, p, s) fractions by one time step.

    The physics:
      1. Energy of playing each strategy, given the current population mix.
         U_strategy = k * (row of the payoff matrix) . (population fractions)
         The payoff matrix is  P = [[ 1, -e,  e],
                                    [ e,  1, -e],
                                    [-e,  e,  1]].
         Diagonal (+1) rewards matching neighbours  -> drives ORDER.
         Off-diagonal (+-e) is the RPS cycle         -> drives CYCLING.
      2. Glauber rule: the rate of switching strategy a -> b is
         w = 0.5 / (1 + exp(-(U_b - U_a)/T)).  Higher-energy target = more likely.
      3. Master equation: new fraction = (what stays) + (inflow) - (outflow).
         This is a forward-Euler step (dt = 1) of the mean-field ODE.
    """
    # 1. interaction energies
    U_r = k * ( r - eps * p + eps * s)
    U_p = k * ( eps * r + p - eps * s)
    U_s = k * (-eps * r + eps * p + s)

    # 2. Glauber transition rates between every ordered pair of strategies
    def w(U_from, U_to):
        return 0.5 / (1.0 + np.exp(-(U_to - U_from) / T))

    w_RP, w_RS = w(U_r, U_p), w(U_r, U_s)
    w_PR, w_PS = w(U_p, U_r), w(U_p, U_s)
    w_SR, w_SP = w(U_s, U_r), w(U_s, U_p)

    # 3. master equation (conserves r + p + s = 1)
    r_next = r * (1 - w_RP - w_RS) + p * w_PR + s * w_SR
    p_next = p * (1 - w_PR - w_PS) + r * w_RP + s * w_SP
    s_next = 1.0 - r_next - p_next
    return r_next, p_next, s_next


def run(eps, k=10.0, T=0.65, steps=4000, init=(0.4, 0.35, 0.25)):
    """Integrate the dynamics and return the time series of fractions."""
    r, p, s = init
    hist = np.empty((steps, 3))
    for t in range(steps):
        hist[t] = (r, p, s)
        r, p, s = step(r, p, s, eps, k, T)
    return hist


def order_parameter(hist, burn_in_frac=0.5):
    """The complex RPS order parameter m_psi.

    Map R, P, S to three unit vectors 120 degrees apart in the complex plane:
        psi(t) = r + p*exp(i 2pi/3) + s*exp(i 4pi/3)
    Then m_psi = | time-average of psi |, measured after a burn-in.

      * Static consensus  -> psi sits at one corner -> |average| ~ 1  (ORDERED)
      * Cycling           -> psi orbits the centre  -> |average| ~ 0  (DISORDERED)

    The magic is the *time average*: a rotating vector averages to zero, so a
    single number cleanly separates the two phases.
    """
    r, p, s = hist[:, 0], hist[:, 1], hist[:, 2]
    psi = r + p * np.exp(1j * 2 * np.pi / 3) + s * np.exp(1j * 4 * np.pi / 3)
    start = int(len(hist) * burn_in_frac)
    return np.abs(np.mean(psi[start:])), np.abs(psi)  # m_psi, |psi(t)|


def main():
    cases = [("Low epsilon (ordering)", 0.2), ("High epsilon (cycling)", 0.9)]
    fig, axes = plt.subplots(2, len(cases), figsize=(11, 7), sharex=True)

    for col, (title, eps) in enumerate(cases):
        hist = run(eps)
        m_psi, psi_mag = order_parameter(hist)

        # top row: the three strategy fractions over time
        ax = axes[0, col]
        ax.plot(hist[:, 0], label="rock",     color="tab:red")
        ax.plot(hist[:, 1], label="paper",    color="tab:green")
        ax.plot(hist[:, 2], label="scissors", color="tab:blue")
        ax.set_title(f"{title}\n(eps={eps})")
        ax.set_ylabel("fraction")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="upper right", fontsize=8)

        # bottom row: instantaneous |psi| and the headline number m_psi
        ax = axes[1, col]
        ax.plot(psi_mag, color="black", lw=1)
        ax.set_title(f"m_psi = {m_psi:.3f}  ->  "
                     f"{'ORDERED' if m_psi > 0.5 else 'DISORDERED'}")
        ax.set_ylabel("|psi(t)|")
        ax.set_xlabel("time step")
        ax.set_ylim(-0.02, 1.02)

    fig.suptitle("Phase 1: Homogeneous Mean-Field Potts-RPS  (k=10, T=0.65)",
                 fontweight="bold")
    fig.tight_layout()
    out = "hmf_phase1.png"
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")
    for title, eps in cases:
        m_psi, _ = order_parameter(run(eps))
        print(f"  eps={eps}:  m_psi={m_psi:.3f}")


if __name__ == "__main__":
    main()
