"""
Phase 5: Degree-based Mean Field (DMF).

HMF (Phase 1) assumed every node feels the SAME average mix -- it knows only the
scalar <k>. DMF does better: it keeps a separate (r, p, s) state for every degree
class k, and couples them through the network's degree distribution P(k). Hub
nodes (large k) feel interactions more strongly, so on a heterogeneous graph
(like BA) DMF can capture structure that HMF averages away.

This is the Python port of the original repo's `hmf_simulation.cpp` (which,
despite its name, is the DEGREE-based mean field, not the homogeneous one).

The key quantity:
  Theta_s = (1/<k>) * sum_k  k * P(k) * rho_s(k)
          = probability that a randomly chosen NEIGHBOUR plays strategy s.
A neighbour is reached with probability proportional to its degree k -- that
k-weighting is the whole point of DMF, and what HMF throws away.
"""

import numpy as np


def run(pk, eps, T=0.65, steps=4000, seed=0):
    """Evolve the DMF equations.

    pk : dict {degree k -> fraction P(k)} from the actual graph.
    Returns the per-step global (r, p, s) fractions.
    """
    rng = np.random.default_rng(seed)
    degs = np.array(sorted(pk.keys()), dtype=float)      # degree classes present
    Pk = np.array([pk[int(k)] for k in degs])            # P(k) for each
    mean_deg = float(np.sum(degs * Pk))
    nk = len(degs)

    # state[i] = (r, p, s) for degree class degs[i]; random initial mix
    st = rng.random((nk, 3))
    st /= st.sum(axis=1, keepdims=True)

    glob = np.empty((steps, 3))
    for t in range(steps):
        # neighbour-strategy probabilities (k-weighted average over degree classes)
        theta = (degs * Pk) @ st / mean_deg              # length-3: (theta_r, theta_p, theta_s)
        tr, tp, ts = theta

        # interaction energies per degree class (scale with k)
        U_r = degs * (tr - eps * tp + eps * ts)
        U_p = degs * (eps * tr + tp - eps * ts)
        U_s = degs * (-eps * tr + eps * tp + ts)

        def logistic(x):
            return 1.0 / (1.0 + np.exp(-x / T))

        # Glauber rates (0.5 = proposal probability of a given other strategy)
        wRP = 0.5 * logistic(U_p - U_r); wRS = 0.5 * logistic(U_s - U_r)
        wPR = 0.5 * logistic(U_r - U_p); wPS = 0.5 * logistic(U_s - U_p)
        wSR = 0.5 * logistic(U_r - U_s); wSP = 0.5 * logistic(U_p - U_s)

        r, p, s = st[:, 0], st[:, 1], st[:, 2]
        nr = r * (1 - wRP - wRS) + p * wPR + s * wSR
        np_ = p * (1 - wPR - wPS) + r * wRP + s * wSP
        ns = s * (1 - wSR - wSP) + r * wRS + p * wPS
        new = np.stack([nr, np_, ns], axis=1)
        new /= new.sum(axis=1, keepdims=True)
        st = new

        glob[t] = Pk @ st                                # global fractions = P(k)-average
    return glob


def order_parameter(glob, burn_in_frac=0.5):
    r, p, s = glob[:, 0], glob[:, 1], glob[:, 2]
    psi = r + p * np.exp(1j * 2 * np.pi / 3) + s * np.exp(1j * 4 * np.pi / 3)
    start = int(len(glob) * burn_in_frac)
    return np.abs(np.mean(psi[start:]))
