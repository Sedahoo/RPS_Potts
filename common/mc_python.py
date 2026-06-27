"""Pure-Python agent-level Monte Carlo (slow but readable reference).

The C++ engine in drivers/ is the production version; this stays for teaching /
cross-checking. Returns the per-sweep (r, p, s) fractions.
"""

import numpy as np


def run_mc(neighbors, eps, T=0.65, sweeps=1500, seed=0):
    rng = np.random.default_rng(seed)
    N = len(neighbors)
    P = [[1.0, -eps, eps], [eps, 1.0, -eps], [-eps, eps, 1.0]]
    states = rng.integers(0, 3, size=N).tolist()
    fracs = np.empty((sweeps, 3))
    for t in range(sweeps):
        nodes = rng.integers(0, N, size=N)
        proposals = rng.integers(1, 3, size=N)
        accepts = rng.random(size=N)
        for i in range(N):
            n = nodes[i]
            cur = states[n]
            new = (cur + proposals[i]) % 3
            nbrs = neighbors[n]
            if not nbrs:
                continue
            Pn, Pc = P[new], P[cur]
            dU = 0.0
            for m in nbrs:
                sm = states[m]
                dU += Pn[sm] - Pc[sm]
            if accepts[i] * (1.0 + np.exp(-dU / T)) < 1.0:
                states[n] = new
        c0 = states.count(0); c1 = states.count(1)
        fracs[t] = (c0 / N, c1 / N, (N - c0 - c1) / N)
    return fracs
