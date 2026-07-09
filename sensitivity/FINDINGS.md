# Sensitivity study — findings

Companion to `HYPOTHESES.md` (pre-registered expectations; read it first).
Each experiment below is confronted with its hypothesis: confirmed parts,
refuted parts, and what the refutations taught us. Headline: the project's
fixed parameters (T, seeds, grid, sweeps, labels, damage realisation) are
either genuine nuisance parameters or well-inside safe regimes — but two
experiments uncovered NEW physics the headline results had hidden: the
transition is first-order-like (S7), which quantitatively explains the
finite-size drift of eps_c (S3).

## S0 — validation grid (Sec. 2): CONFIRMED, stronger than expected

45/45 (N x eps x seed) pairs give the same phase verdict; max |m_py - m_cpp|
= 0.0054 and falls with N (0.0054 / 0.0049 / 0.0018 for N=100/300/500),
the 1/sqrt(N) self-averaging expected of a global average. The predicted
"expected disagreements inside the transition window" never materialised
because the window (width < 0.03 in eps at these sizes) is narrower than the
0.15 spacing of the tested eps values — no test point landed inside it. No
systematic offset: the two engines are the same process to statistical
accuracy, everywhere probed, not just at the 4 showcase points.

## S7 — mean-field initial composition (Sec. 3): hypothesis PARTLY WRONG,
## and the failure is a discovery

Predicted: biased inits all agree; only the near-symmetric start misbehaves.
Observed: the near-symmetric start misbehaves as predicted (transient dip of
m at eps ~ 0.58, k=10 — it lingers near the unstable symmetric point), BUT
the biased inits do NOT agree near the transition: at k=10, inits
(0.4,.35,.25)/(0.5,.3,.2) give eps_c = 0.631 while (0.9,.05,.05) sustains
order to 0.706; at k=20 the window is [0.706, 0.806]. This is not a
transient artifact (the curves are flat at m ~ 1 vs m ~ 0 inside the window)
— it is genuine BISTABILITY: consensus fixed point and limit cycle are
simultaneously stable, and the init picks the attractor. The HMF transition
is therefore subcritical (first-order-like, hysteretic), not a soft
crossing. Practical consequence: a mean-field "eps_c" is only defined given
an init convention; the production init lies on the conservative (lower)
edge of the window. Cross-link: this is exactly the ingredient that explains
S3's 1/N drift and the sharpness of every MC transition curve in the report.

## S1 — temperature x degree (Sec. 4): MC CONFIRMED; HMF part REFUTED

MC: eps_c falls monotonically with T at every k, and the fall flattens as k
grows — drop from T=0.3 to 1.3 is 0.197 (k=8), 0.073 (k=20), 0.027 (k=40).
Both predictions (noise shrinks the ordered phase; effective noise ~ T/k)
hold; T=0.65 sits in a regime where the k-dependence dominates, which is why
the report can meaningfully present eps_c as a function of <k> alone.
Two-phase structure survives at every (T, k), as predicted.

HMF: the prediction "overestimates eps_c at every (T,k)" is WRONG at high k.
The HMF gap is +0.10..0.17 at k=8, shrinks at k=20, and REVERSES at k=40
(HMF ~ 0.66-0.71 vs MC 0.70-0.73), with a non-monotonic ±0.05 wiggle in T.
Explanation, using S7: for k*U >> T the mean-field transition rates saturate
(the logistic becomes a step), so the map — and its eps_c — becomes nearly
(T,k)-independent and plateaus ~ 0.7, while the MC boundary keeps rising
with k. And inside the bistable window a single-init eps_c measures a basin
boundary, not a bifurcation point, which is why it wiggles instead of moving
smoothly. The report's "HMF sits above MC" claim (Sec. 4.1) is drawn for
k <= 10, where it is correct; this experiment marks its domain of validity.

## S2 — graph seed vs MC seed (Sec. 4): CONFIRMED

25 (graph, MC)-seed combinations: eps_c = 0.6257 +- 0.0025 (total std),
range 0.013 — a factor 20 below the 0.05 grid step. Decomposition: graph-seed
std 0.0010, MC-seed std 0.0010 — indistinguishable, as predicted from the
local homogeneity of ER at <k>=20. m_psi scatter peaks exactly at the
crossing (std 0.066 at eps=0.65) and is zero at eps=0. Single-realisation
results at this size are representative; seed averaging buys cosmetics, not
correctness.

## S5 — eps-grid resolution (Sec. 4): CONFIRMED

Same data, four grids. Interpolated estimator: shifts of +0.005/-0.005/+0.020
for steps 0.025/0.05/0.10 — all under half a step, under a QUARTER of a step
at the production resolution 0.05. Naive first-point-below estimator: +0.020
at step 0.05 and +0.070 at step 0.10 — bias of order the full step, always
positive (it can only round up). The project-wide switch to the interpolated
estimator is worth ~ 4x in effective resolution.

## S3 — system size (Sec. 5): half CONFIRMED, half corrected by the data

Confirmed: cycling-phase m_psi ~ 1/sqrt(N) over 5 octaves (0.0040 at N=125
to 0.0006 at N=4000 ~ 1/sqrt(32) = 0.18x); seed scatter shrinks similarly;
the transition sharpens (but the width metric saturates at seed-noise level
~ 0.01 for N >= 500 — 4 seeds cannot resolve it further).

Corrected: "eps_c settles by N ~ 1000" is wrong. On the fine grid the
crossing drifts DOWN smoothly: 0.700, 0.665, 0.629, 0.610, 0.594, 0.587 for
N = 125..4000 — and (eps_c(N) - 0.58) halves per doubling of N: the shift is
~ 1/N, extrapolating (Richardson, last two sizes) to eps_c(inf) = 0.580 at
k=20. A 1/N shift of the pseudo-transition point is the STANDARD finite-size
scaling of a FIRST-ORDER transition — independent confirmation of S7's
subcriticality from a completely different direction (MC finite-size scaling
vs mean-field bistability). Consequence for the report: quoted eps_c values
are N=500 finite-size estimates, biased high by ~ +0.05 at k=20; every
COMPARISON in the report (ER vs BA, damaged vs pristine, MC vs MF) is made
at matched N, so the conclusions are unaffected, but absolute eps_c values
should be read with that offset in mind.

## S4 — simulation length (Sec. 5): mechanism half-right, sign WRONG,
## default validated

Predicted overestimate of eps_c at short sweeps (cycling-side up-bias).
Observed: eps_c = 0.599 at 200 sweeps — an UNDERestimate — converging to
0.625 by 1000 sweeps and flat thereafter (0.6252/0.6252/0.6250/0.6268).
Both predicted biases exist, but their magnitudes were misjudged: the
cycling-side up-bias is real yet tiny in absolute terms (m_cycle = 0.0055 at
200 sweeps vs 0.0008 at 6000 — nowhere near the 0.5 threshold), while the
ordered-side down-bias is large (140 measured sweeps cannot finish ordering
near the transition, so m dips below 0.5 early and the crossing moves LEFT).
The crossing sits on the steep ordered flank, so the ordered-side bias owns
the sign. RMSE against the 6000-sweep reference: 0.113 at 200 sweeps, 0.030
at 500, ~ 0.014 from 1000 on (residual seed noise, not drift). The
production default (1500) sits safely on the plateau with ~ 2x margin.

## S6 — zealot strategy label (Sec. 6): CONFIRMED, with a bonus

With 32 seeds and a paired permutation test (labels are exchangeable within
a seed under the null — the correct test, since seed-level outcomes are
bimodal at large z where a realisation either flips to the zealot consensus
or not): ordering phase 0/8 z-points significant at 0.05, cycling phase 1/8
marginal (p=0.02) — 1/16 overall, the textbook false-positive rate. The
backfire effect appears for every label (each label's free network converges
onto that label's beater, dashed curves). Bonus discovery: at intermediate z
in the ordering phase the label spread is EXACTLY zero — relabeled runs
COALESCE. The engine's RNG consumption is state-independent, so the three
relabeled runs share an aligned random stream; once each reaches its (label-
rotated) consensus they are the same trajectory up to rotation, and every
label-invariant observable agrees to machine precision (verified directly:
(r,p,s) of the three runs are exact cyclic permutations). A per-run exact
symmetry check, far stronger than the statistical one — and free.

## S8 — damage realisation (Sec. 6): CONFIRMED

Which edges are removed does not matter. First, effective degree does not
even fluctuate: remove_edges cuts a deterministic COUNT of edges and the
giant component stays whole at these densities, so k_eff = 17.00/14.00/11.00
exactly, across all 6 damage seeds. Second, eps_c scatter across seeds is
0.007/0.0001/0.012 for f=0.15/0.30/0.45 — a quarter of a grid step; the
residual comes from which edges die, and it is negligible. Third, the
seed-mean eps_c lands on the pristine ER boundary interpolated at k_eff
(deviations +0.036/-0.005/+0.010, all within the boundary's own grid
resolution) — Sec. 6.5's collapse holds realisation by realisation, not just
on average. The mild growth of scatter with f matches the hypothesis.

## Synthesis

1. Nuisance parameters verified nuisance: graph seed, MC seed, eps-grid step
   (with the interpolated estimator), zealot label, damage realisation.
   Each was TESTED, not assumed — and each null result has a stated
   mechanism (self-averaging, exact symmetry, deterministic edge count).
2. Regime parameters validated: 1500 sweeps (2x margin), N=500
   (representative, with a quantified +0.05 finite-size offset at k=20),
   T=0.65 (inside the k-dominated regime that makes "eps_c(<k>)" the right
   axis).
3. New physics from the failures: the order-cycling transition is
   first-order-like — mean-field bistability window (S7), 1/N shift of the
   MC pseudo-transition (S3), and the step-like m(eps) curves all say the
   same thing. The HMF is quantitative only at low k; by k=40 its saturated
   rates undershoot the MC boundary (S1).
