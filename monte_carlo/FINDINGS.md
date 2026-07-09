# Findings — monte_carlo/

The stochastic ground truth: agents on a real network, plus validation of the
C++ engine against the pure-Python reference.

## MC dynamics (`mc.py`)

- On an ER graph (N~500, <k>=10): eps=0.2 orders, `m_psi=0.997`; eps=0.9 cycles,
  `m_psi=0.001`. Matches HMF in both phases.
- **Finite-size physics the mean field can't see:** in the cycling phase, the
  deterministic HMF shows a *coherent* global oscillation (instantaneous |psi|
  stays high, only the time-average cancels), but the finite-N MC shows *noisy
  coexistence* near the centre — instantaneous |psi| is also small. Both give
  `m_psi ~ 0`, but the mechanism differs. (This reproduces the discrepancy on
  the thesis "order parameter" slide between the MC and mean-field panels.)

## MC vs HMF (`compare.py`)

- They agree deep in each phase but **diverge at the transition**: MC breaks at
  eps_c ~ 0.53, HMF not until ~0.62. Mean-field theory overestimates the ordered
  phase because it neglects fluctuations. This single plot is why the project
  keeps both models.

## Engine validation (`validate_engine.py`)

- The C++ engine matches the pure-Python MC within stochastic noise (different
  RNGs; ordered/disordered verdict always agrees) and runs **~30-40x faster**.
  This is what makes the parameter sweeps in `phase_diagram/` feasible.

## Overlay grid across T, <k>, N + BA (`compare_grid.py`)
The MC-vs-HMF overlay repeated with the validated C++ engine over T/k/N plus
a BA row. MC–HMF eps_c gap shrinks with k (0.16 → 0.08 from k=6 to 20), does
not grow with T (both boundaries slide together at k=10), and widens slowly
with N (the MC crossing walks left ~1/N under a frozen HMF). BA at matched
<k> reproduces ER to line width — the mean-field error is set by <k> and N,
not by P(k).
