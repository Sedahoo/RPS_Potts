# Sensitivity study — pre-registered hypotheses

Written BEFORE running any experiment in this folder. Each report section gets
its own robustness experiments: we identify the parameters that section holds
fixed, vary each over a grid, state the expected outcome and mechanism, and
name the metrics that should move and the ones that should not. No parameter
is assumed unimportant — the "should not matter" cases (seeds, zealot label,
grid step) are tested exactly like the physical ones. `FINDINGS.md` (written
after the runs) confronts each hypothesis with the data; the results are
embedded in the matching section of RESULTS_REPORT.pdf.

Shared conventions: ER graphs unless stated, engine defaults (T=0.65,
1500 sweeps, burn-in 30%), eps grid 0..1 step 0.05 unless stated, eps_c =
interpolated m_psi=0.5 crossing (the project-wide estimator,
`common.observables.eps_crossing`). "Baseline" = N=500, <k>=20, T=0.65.

---

## Report Sec. 2 — Engine validation

### S0 — Does C++/Python agreement hold across the parameter space? (`sens_validation.py`)

The headline validation is 4 eps values on ONE graph with ONE seed pair. Grid
here: N in {100, 300, 500} x eps in {0.2, 0.35, 0.5, 0.65, 0.8} x 3 seeds,
k=10 — 45 paired runs (same graph and seed through both engines).

**Hypothesis.** The two engines implement the same stochastic process with
different RNG streams, so they must agree in DISTRIBUTION, not per-run:
same-side-of-0.5 verdicts everywhere except possibly at (N, eps) points that
sit within the transition region, where m ~ 0.5 and two independent runs can
land on opposite sides — such a disagreement is expected noise, not a bug,
and must vanish away from the transition. |m_py - m_cpp| should grow with
proximity to eps_c and shrink with N (fluctuations of a global average ~
1/sqrt(N)). A systematic offset (one engine always higher) at ANY grid point
away from the transition would indicate an implementation difference.

## Report Sec. 3 — Mean field

### S7 — Initial composition of the HMF map (`sens_mf_init.py`)

Grid: init (r,p,s)_0 in {(0.3334, 0.3333, 0.3333) near-symmetric,
(0.40, 0.35, 0.25) production default, (0.50, 0.30, 0.20), (0.90, 0.05, 0.05)}
x k in {10, 20} x 81 eps values; 4000 map steps, m over the last half.

**Hypothesis.** The HMF map is deterministic, so what matters is which
attractor the start flows to and how fast. Deep in either phase the attractor
(consensus fixed point / limit cycle) is reached quickly: m(eps) and eps_c
should be init-independent there — the fixed point reached does not depend on
how strongly the start leans toward it, so the biased inits (0.5.., 0.9..)
should agree with the default everywhere. The near-symmetric start is the
exception: (1/3,1/3,1/3) is an UNSTABLE fixed point of the map, so a start
1e-4 away lingers near it for a transient ~ log(1/delta)/(growth rate); where
that transient is slow (near eps_c, critical slowing down) the measurement
window may close before the attractor is reached, misreading the phase and
shifting the apparent eps_c for that init only.

## Report Sec. 4 — Phase diagram

### S1 — Temperature T x average degree k (`sens_temperature.py`)

The whole phase diagram is taken at one temperature. Grid: T in {0.30, 0.50,
0.65, 0.80, 1.00, 1.30} x k in {8, 20, 40} x 4 graph seeds x 21 eps; plus the
HMF prediction eps_c(T, k) on the same grid.

**Hypothesis.** T is a genuine control parameter: raising it shrinks the
ordered phase, so eps_c should DECREASE monotonically with T at every k.
Mechanism: Glauber acceptance 1/(1+e^{-dU/T}) compares the payoff gain to T;
more noise means the ferromagnetic (identity) part of the payoff pins
consensus less effectively, so less cyclic drive eps is needed to destabilise
it. BUT the payoff a node feels scales with its degree (dU ~ k), so the
effective noise is ~T/k: the T-dependence of eps_c should be visibly WEAKER
at k=40 than at k=8 (a T-k interaction), i.e. the report's headline
"stability is a function of <k>" is the k-dominated regime of a 2-parameter
surface. Within the tested range T stays far below the eps=0 Potts ordering
temperature (scale ~ k), so eps_c > 0 everywhere. HMF should reproduce the
trend but overestimate eps_c at every (T, k) since it ignores fluctuations.
Unaffected: the two-phase structure itself.

### S2 — Graph seed vs MC seed (`sens_seeds.py`)

Grid: 5 graph seeds x 5 engine seeds (25 combinations) x 21 eps, baseline.

**Hypothesis.** Both seeds are nuisance parameters at N=500: eps_c should
scatter by well under one grid step (0.05) across all 25 combinations, and
m_psi scatter should peak near eps_c (steep response / critical fluctuations)
and be negligible deep in either phase. Graph-seed and MC-seed scatter should
be COMPARABLE, because an ER graph at <k>=20 is locally homogeneous — degree
fluctuations sqrt(k)/k ~ 22% give no realisation special structure. If graph
scatter dominated, single-realisation results in the report would not be
representative — that is the assumption being audited.

### S5 — eps-grid resolution (`sens_grid.py`)

One fine sweep (eps step 0.0125, 81 points) x 4 graph seeds at baseline;
coarser grids (steps 0.025, 0.05, 0.10) by exact subsampling, so every
resolution sees identical simulation data and only the estimator's grid
changes.

**Hypothesis.** The interpolated estimator makes grid step a nuisance
parameter: m(eps) is close to linear across one step near the crossing, so
linear interpolation should keep the eps_c shift well below half a step even
at step 0.10, converging as the step shrinks. The naive estimator (first grid
point with m < 0.5 — the project's old convention) should be biased by up to
a full step, quantifying what interpolation buys.

## Report Sec. 5 — Dynamics and finite-size scaling

### S3 — System size N (`sens_size.py`)

Grid: N in {125, 250, 500, 1000, 2000, 4000} x 4 graph seeds x 21 eps, k=20.

**Hypothesis.** eps_c is a property of the local neighbourhood (<k>), not of
N: the crossing should drift only weakly and settle by N ~ 1000 (extending
the report's FSS section, which is at k=10, to the k=20 operating point).
Three things SHOULD change with N: (i) the transition width (distance between
the m=0.75 and m=0.25 crossings) shrinks as the crossover sharpens toward a
step; (ii) seed-to-seed scatter shrinks ~ 1/sqrt(N) (self-averaging of a
global average); (iii) cycling-phase m_psi decreases with N, because finite-N
cycling amplitude is a fluctuation effect.

### S4 — Simulation length (sweeps) (`sens_equilibration.py`)

Grid: sweeps in {200, 500, 1000, 1500, 3000, 6000} (burn-in fixed at 30%) x 4
graph seeds x 21 eps, baseline. Reference = the 6000-sweep curve.

**Hypothesis.** Two opposite finite-TIME biases on opposite sides of the
transition. Cycling side: m_psi = |time-average of psi| vanishes only if the
window covers many full rotations of psi(t); a short window averages a
fraction of a cycle, biasing m_psi UP — the cycling phase looks partially
ordered. Ordered side: a short run may not have escaped the initial
disordered state, biasing m_psi DOWN. The cycling-side up-bias moves the
m=0.5 crossing right, so eps_c should be OVERestimated at small sweeps and
converge from above by ~1500 (the production default — this audits that
choice). RMSE against the 6000-sweep reference should fall monotonically.

## Report Sec. 6 — Perturbation experiments

### S6 — Zealot strategy label (`sens_zealot_symmetry.py`)

Grid: zealot strategy in {Rock, Paper, Scissors} x 8 zealot fractions
(0.025..0.20) x 32 graph seeds x both phases (eps=0.3, eps=0.9), N=800, k=10,
random placement (matching Sec. 6.1). The graph seed also seeds the zealot
placement, so placement realisation is varied at the same time.

**Hypothesis.** NO effect: the payoff matrix is exactly symmetric under the
cyclic relabeling R->P->S->R, so conversion(z) and m_psi(z) must be
statistically identical for the three labels — label differences within the
seed-to-seed scatter at every z. The backfire effect (zealots hand the free
network to the strategy that BEATS them) must appear for every label:
Paper-zealots breed Scissors, etc. This is the designated
"parameter-that-should-not-matter" check: the symmetry is exact in the model
definition, so any systematic label dependence would expose an implementation
bug (asymmetric proposal move, payoff indexing), not physics.

### S8 — Damage realisation seed (`sens_defect_seed.py`)

Grid: 6 damage seeds x damage fraction f in {0.15, 0.30, 0.45} (edge
removal) x 21 eps on one ER graph (N=500, pristine <k>=20), engine defaults.

**Hypothesis.** WHICH edges are removed should not matter, only HOW MANY:
Sec. 6.5's collapse says damaged networks sit on the pristine eps_c(<k>)
curve, so eps_c across damage seeds at fixed f should scatter no more than
the tiny scatter of the surviving effective degree 2E'/N' — and plotting
eps_c against effective <k> should put all 18 points on the pristine
boundary regardless of seed. Scatter should grow mildly with f (fewer
surviving edges, relatively larger realisation differences).

---

## Amendments after the first pass (methods, not hypotheses)

The hypotheses above were not changed after seeing data; three measurement
protocols were, and the reasons are recorded here.

1. **S6**: the original grid started at z=0 with 8 seeds and compared the
   label spread to 2x the mean seed std. z=0 is degenerate (with no zealots
   all three labels measure the SAME run against different names, so their
   "spread" is deterministic, not statistical) and the Gaussian criterion is
   miscalibrated where outcomes are bimodal across seeds (a realisation
   either flips to the zealot consensus or not). Fixed: z starts at 0.025,
   32 seeds, and a paired permutation test (labels exchangeable within each
   seed under the null) replaces the 2-sigma rule.
2. **S3**: eps_c and width are now estimated on a fine zoom grid (step 0.0125
   across the transition); the 0.05 production grid quantised the N-drift
   into a staircase because the transition at k=20 is sharper than one grid
   step.
3. **S1**: the HMF eps_c uses a fine eps grid (step 0.01) — the map is
   deterministic, so its crossing has no business being quantised by the MC
   grid.
