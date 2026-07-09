"""The RPS order parameter, shared by the MC and the mean-field models."""

import numpy as np

# the three RPS strategies as unit vectors 120 degrees apart in the complex plane
_PHASE = np.array([1.0, np.exp(1j * 2 * np.pi / 3), np.exp(1j * 4 * np.pi / 3)])


def psi_series(fracs):
    """Complex order-parameter signal psi(t) from a time series of (r, p, s)."""
    fracs = np.asarray(fracs, dtype=float)
    return fracs @ _PHASE


def order_parameter(fracs, burn_in_frac=0.5):
    """m_psi = | time-average of psi |, measured after a burn-in.

    ~1 for static consensus (ordered), ~0 for cycling (the rotating vector
    averages to zero).
    """
    psi = psi_series(fracs)
    start = int(len(psi) * burn_in_frac)
    return float(np.abs(np.mean(psi[start:])))


def eps_crossing(eps, m, thr=0.5):
    """Interpolated eps where m(eps) first crosses below thr (order -> cycling).

    The project-wide eps_c estimator (same as phase_diagram/critical_boundary.py):
    sort by eps, find the first grid point with m < thr, linearly interpolate
    between the bracketing points.
    """
    eps = np.asarray(eps, dtype=float)
    m = np.asarray(m, dtype=float)
    o = np.argsort(eps)
    eps, m = eps[o], m[o]
    below = np.where(m < thr)[0]
    if len(below) == 0:
        return float(eps[-1])
    i = below[0]
    if i == 0:
        return float(eps[0])
    e0, e1, m0, m1 = eps[i - 1], eps[i], m[i - 1], m[i]
    return float(e0 + (thr - m0) * (e1 - e0) / (m1 - m0))
