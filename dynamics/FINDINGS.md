# Findings — dynamics/

Three views of the dynamics beyond a single order parameter: the state's path on
the simplex, how the transition scales with system size, and the stability of
the symmetric fixed point.

## Ternary simplex (`ternary.py`)

- Every (r,p,s) is a point in a triangle with pure-strategy corners. **Low eps**:
  the path walks into a corner (consensus). **High eps**: a dense cloud orbiting
  the centre (cyclic coexistence). The two phases, made geometric.

## Finite-size scaling (`fss.py`)

- The `m_psi(eps)` curve is **rounded and shifted late** on small networks
  (N=200: eps_c ~ 0.55, gradual) and **sharpens + converges** as N grows
  (N=1000/2000: near-vertical drop at eps_c ~ 0.50). Both the rounding and the
  finite-size shift of the apparent critical point are textbook critical-
  phenomena signatures — a true sharp transition exists only as N -> infinity.

## Fixed-point stability (`stability.py`)

- The mixed state (1/3,1/3,1/3) is always a fixed point; the transition is really
  about ITS stability. Nudging it by delta=0.02 and integrating the HMF:
  **low eps (0.2, 0.5)** -> collapses to a corner (consensus); **high eps
  (0.7, 0.95)** -> blooms into a sustained limit cycle (larger amplitude at
  higher eps). A Hopf-like change in stability *is* the transition.
