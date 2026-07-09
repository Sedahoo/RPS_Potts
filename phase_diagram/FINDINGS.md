# Findings — phase_diagram/

The headline figure: a heatmap of `m_psi` over (average degree `<k>`, cyclic
strength `eps`), from the full stochastic MC. Reproduces the thesis's
"Connectivity vs Stability" result.

## The phase boundary (`run.sh` -> `plot_phase_diagram.py`)

- Red (ordered, `m_psi`~1) fills the upper-left; blue (cycling, `m_psi`~0) the
  lower-right. The boundary **curves up and to the right**: as `<k>` grows, order
  survives to higher `eps`. Connectivity protects order — now from the real
  simulator, not the mean-field toy.
- The boundary **saturates near eps ~ 0.65** at high `<k>` (consistent with the
  HMF `T/k` saturation in `mean_field/`).
- The bottom rows (`<k>` = 2-4, where the graph is barely connected) **cannot
  sustain order even at small eps** — a finite-size / percolation effect the mean
  field misses.

## ER vs BA

- The ER and BA phase diagrams are **nearly identical**. For the MC dynamics what
  matters is the *average* degree, not whether the distribution is homogeneous
  (ER) or hub-dominated (BA). (The degree distribution's effect shows up instead
  in how well DMF vs HMF approximate the MC — see `mean_field/`.) BA's boundary
  is slightly fuzzier due to degree heterogeneity.

## The extracted boundary (`critical_boundary.py`)

- Pulling eps_c(<k>) out of both heatmaps (interpolated m_psi=0.5 crossing) and
  overlaying the HMF prediction gives the whole story in one curve:
  **max |eps_c(ER) - eps_c(BA)| = 0.040** across all 20 degrees (they coincide),
  and the HMF prediction sits **above** the MC boundary at every k (mean field
  overestimates the ordered phase). Data: `critical_boundary.csv`.

## Extra diagrams (`extra_diagrams.py`)
Four more (⟨k⟩×ε) diagrams around the production point (T=0.30/1.00 at N=800,
N=300/2000 at T=0.65). The two-phase structure is universal; only the boundary
moves: up when cold (+0.024 mean), down when hot (−0.028), with the T-shift
concentrated at small <k> (T/k effective noise); size shifts smaller and
signed as the 1/N first-order drift (+0.040 at N=300, −0.017 at N=2000).
