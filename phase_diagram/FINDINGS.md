# Findings — phase_diagram/

The headline figure: a heatmap of `m_psi` over (average degree `<k>`, cyclic
strength `eps`), from the full stochastic MC. Reproduces the thesis's
"Connectivity vs Stability" result.

## The phase boundary (`run.sh` -> `plot_phase_diagram.py`)

- Red (ordered, `m_psi`~1) fills the upper-left; blue (cycling, `m_psi`~0) the
  lower-right. The boundary **curves up and to the right**: as `<k>` grows, order
  survives to higher `eps`. Connectivity protects order — now from the real
  simulator, not the mean-field toy.
- The boundary rises more and more slowly at high `<k>` (~0.66 at k=24, ~0.74
  by k=80 on the grid extended to k=80) — a slow creep, not a hard saturation:
  it is the *HMF* that hard-saturates (`T/k` rate saturation in `mean_field/`),
  which is exactly why the MC overtakes it (see the extracted boundary below).
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
  overlaying the HMF prediction — recomputed at **every** MC degree (k=2..80) on
  the identical eps grid with the identical estimator — gives the whole story in
  one curve: **max |eps_c(ER) - eps_c(BA)| = 0.040** across all 40 degrees (they
  coincide). The plotted HMF curve is a standard-init sweep; because the
  mean-field transition is subcritical, a second (ordered-init) sweep is also
  computed to bound the other edge of the bistable window (not plotted, table
  only). Result: the standard-init edge sits above the MC at low k (+0.198 at
  k=4) but dips *below* it past k≈40 — this is **not a mean-field failure**,
  it's the MC boundary entering the widening window: at k=80 the window is
  [0.70, 0.94] and the MC boundary runs *inside* it for 21/40 degrees (k≥40),
  where a single-init eps_c stops being a unique prediction. Data:
  `critical_boundary.csv` (`eps_c_HMF` = standard init, `eps_c_HMF_ordered_init`
  = the window's other edge).

## Extra diagrams (`extra_diagrams.py`)
Four more (⟨k⟩×ε) diagrams around the production point (T=0.30/1.00 at N=800,
N=300/2000 at T=0.65). The two-phase structure is universal; only the boundary
moves: up when cold (+0.024 mean), down when hot (−0.028), with the T-shift
concentrated at small <k> (T/k effective noise); size shifts smaller and
signed as the 1/N first-order drift (+0.040 at N=300, −0.017 at N=2000).
