"""Deterministic mean-field engines: HMF (homogeneous) and DMF (degree-based).

Both return a per-step time series of the GLOBAL (r, p, s) fractions; callers
turn that into m_psi via common.observables.order_parameter.
"""

import numpy as np


# --------------------------------------------------------------------------- #
# Homogeneous mean field: one (r, p, s), every node feels the average mix.
# --------------------------------------------------------------------------- #
def hmf_step(r, p, s, eps, k, T):
    U_r = k * (r - eps * p + eps * s)
    U_p = k * (eps * r + p - eps * s)
    U_s = k * (-eps * r + eps * p + s)

    def w(U_from, U_to):
        return 0.5 / (1.0 + np.exp(-(U_to - U_from) / T))

    wRP, wRS = w(U_r, U_p), w(U_r, U_s)
    wPR, wPS = w(U_p, U_r), w(U_p, U_s)
    wSR, wSP = w(U_s, U_r), w(U_s, U_p)
    r_n = r * (1 - wRP - wRS) + p * wPR + s * wSR
    p_n = p * (1 - wPR - wPS) + r * wRP + s * wSP
    return r_n, p_n, 1.0 - r_n - p_n


def hmf_run(eps, k=10.0, T=0.65, steps=4000, init=(0.4, 0.35, 0.25)):
    r, p, s = init
    hist = np.empty((steps, 3))
    for t in range(steps):
        hist[t] = (r, p, s)
        r, p, s = hmf_step(r, p, s, eps, k, T)
    return hist


# --------------------------------------------------------------------------- #
# Degree-based mean field: a separate (r, p, s) per degree class, coupled
# through the network's degree distribution P(k).
# --------------------------------------------------------------------------- #
def dmf_run(pk, eps, T=0.65, steps=4000, seed=0):
    """pk: dict {degree -> P(k)}. Returns per-step global (r, p, s)."""
    rng = np.random.default_rng(seed)
    degs = np.array(sorted(pk.keys()), dtype=float)
    Pk = np.array([pk[int(k)] for k in degs])
    mean_deg = float(np.sum(degs * Pk))

    st = rng.random((len(degs), 3))
    st /= st.sum(axis=1, keepdims=True)

    glob = np.empty((steps, 3))
    for t in range(steps):
        tr, tp, ts = (degs * Pk) @ st / mean_deg     # neighbour-strategy probs
        U_r = degs * (tr - eps * tp + eps * ts)
        U_p = degs * (eps * tr + tp - eps * ts)
        U_s = degs * (-eps * tr + eps * tp + ts)

        def logistic(x):
            return 1.0 / (1.0 + np.exp(-x / T))

        wRP = 0.5 * logistic(U_p - U_r); wRS = 0.5 * logistic(U_s - U_r)
        wPR = 0.5 * logistic(U_r - U_p); wPS = 0.5 * logistic(U_s - U_p)
        wSR = 0.5 * logistic(U_r - U_s); wSP = 0.5 * logistic(U_p - U_s)
        r, p, s = st[:, 0], st[:, 1], st[:, 2]
        new = np.stack([r * (1 - wRP - wRS) + p * wPR + s * wSR,
                        p * (1 - wPR - wPS) + r * wRP + s * wSP,
                        s * (1 - wSR - wSP) + r * wRS + p * wPS], axis=1)
        new /= new.sum(axis=1, keepdims=True)
        st = new
        glob[t] = Pk @ st
    return glob
