# Findings — mean_field/

Analytical models: HMF (homogeneous, one (r,p,s)) and DMF (degree-based, one
(r,p,s) per degree class). Reproduces the thesis's analytical results.

## HMF: the order <-> cycling transition (`hmf.py`, `sweep.py`)

- A single control knob `eps`: at eps=0.2 the population orders, `m_psi=1.000`;
  at eps=0.9 it cycles forever, `m_psi=0.001`. The order parameter cleanly
  separates the two phases.
- Sweeping eps at several average degrees `k`, the transition slides to higher
  eps as `k` grows: k=2 breaks at eps~0.08, k=10 at ~0.64, k=200 at ~0.70.
  **Connectivity stabilises order.** Mechanism: every energy is `U = k*(...)`,
  so larger k acts like a lower effective temperature `T/k`.
- The transition saturates at high k (k=10/50/200 all land ~0.64-0.70) — once
  `T/k` is small the dependence flattens.

## DMF and the MC/HMF/DMF comparison (`compare_suite.py`)

- All three models agree in the bulk (ordered at low eps, cycling at high eps).
- Both mean fields **overestimate the ordered phase**: MC breaks at eps~0.50 but
  HMF/DMF hold order to eps~0.62. Mean field ignores the fluctuations that
  destroy order early on a finite network. The RMSE(model, MC) is concentrated
  entirely in this transition band.
- **DMF beats HMF on both graphs, by more on BA** (heterogeneous P(k)):
  RMSE(HMF,MC)/RMSE(DMF,MC) = 0.3376/0.3340 on ER, 0.3357/0.3280 on BA.
  The degree-resolved model earns its keep where the degree spread is wide.
