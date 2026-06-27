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
