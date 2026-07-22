# Findings

Results from the whole project, in two parts: **A. Recreation** (reproducing the
thesis's Project B) and **B. Iteration** (novel extensions). Each section folder
has its own `FINDINGS.md` with the detailed version; this file is the summary.

**Model recap.** Nodes play Rock/Paper/Scissors; payoff `P = I + eps*skew`. `eps`
is cyclic-dominance strength; `m_psi` ~1 ordered (consensus), ~0 cycling. Cycle:
Paper beats Rock, Scissors beats Paper, Rock beats Scissors. Central question:
does higher connectivity `<k>` protect order against `eps`?

---

## A. Recreation findings (reproducing the thesis)

### Mean field — `mean_field/FINDINGS.md`
- One control knob: eps=0.2 orders (`m_psi=1.000`), eps=0.9 cycles (`m_psi=0.001`).
- **Connectivity stabilises order**: the transition slides to higher eps as `<k>`
  grows (k=2 ~0.08, k=10 ~0.64, k=200 ~0.70), via the effective temperature `T/k`;
  it saturates at high k.
- Both mean fields **overestimate the ordered phase** (MC breaks ~0.50, mean field
  ~0.62). **DMF beats HMF**, by more on heterogeneous BA
  (RMSE 0.3357/0.3280 BA vs 0.3376/0.3340 ER).

### Monte Carlo — `monte_carlo/FINDINGS.md`
- MC matches HMF in the bulk but **diverges at the transition** (MC eps_c ~ 0.53
  vs HMF ~ 0.62) — finite-size fluctuations destroy order early.
- **Finite-N signature**: cycling shows noisy coexistence (small instantaneous
  |psi|), unlike the mean field's coherent oscillation — same `m_psi~0`, different
  mechanism.
- C++ engine validated against pure Python, **~30-40x faster** (enables the sweeps).

### Phase diagram — `phase_diagram/FINDINGS.md`
- The `(<k>, eps)` heatmap (grid extended to `<k>`=80): ordered region in the
  upper-left, boundary **curving up-right** (connectivity protects order),
  creeping slowly from ~0.66 (k=24) to ~0.74 (k=80); sparse rows (`<k>`<=4)
  cannot order at all.
- **ER vs BA nearly identical** — for the MC, average degree matters, not the
  degree-distribution shape. Quantified by `critical_boundary.py`:
  max |eps_c(ER) − eps_c(BA)| = 0.040 over all 40 degrees. The HMF prediction,
  recomputed at every MC degree on the identical eps-grid/estimator, sits above
  the MC boundary by +0.198 at k=4 (standard-init edge) but dips below it past
  k≈40 — not a mean-field failure: because the transition is subcritical, the
  HMF is really a bistable *window* (standard-init and ordered-init edges), and
  the window widens with k until the MC boundary runs inside it (21/40 degrees,
  k≥40; window [0.70, 0.94] at k=80) — a single-init eps_c stops being a unique
  prediction there.

### Dynamics — `dynamics/FINDINGS.md`
- **Ternary**: low eps spirals to a corner (consensus); high eps orbits the centre.
- **FSS**: the transition is rounded/late on small N and sharpens + converges to
  eps_c ~ 0.50 as N grows (textbook finite-size scaling).
- **Stability**: the mixed fixed point loses stability across the transition — to
  consensus at low eps, to a limit cycle at high eps (Hopf-like).

---

## B. Iteration findings (novel — not in the thesis)

### Zealots — `zealots/FINDINGS.md`
1. **Zealots provoke their own predator**: a few Rock-zealots flip the free
   ordering-phase network to **Paper**, not Rock; conversion is non-monotonic with
   a frustration minimum near z~0.16.
2. **Hubs amplify zealots ~8x** (cycling-phase `m_psi` 0.72 vs 0.08 random) — but
   the order is still Paper. Hubs control *whether*, not *what*. Ordering-phase
   crossover: provoke Paper at small z, pin Rock once z>0.08.
3. **Competing Rock+Paper factions** -> the **predator wins** (Paper) at high z;
   intermediate z is frustrated/multistable; the cycling phase is robust.

### Defects — `defects/FINDINGS.md`
4. **Defects erode order via effective `<k>`**: edge or node quenching slides the
   transition to lower eps (f=0 -> eps_c~0.63, f=0.8 -> ~0.22), and **edge and node
   defects coincide when matched by the resulting `<k>`** — "connectivity
   stabilises order" run in reverse. The collapse test (`collapse.py`) makes it
   exact: all damaged-network (eps_c, <k>) points land on the pristine-ER
   boundary curve; max edge-node gap 0.014.

### Sensitivity suite — `sensitivity/FINDINGS.md`
Hypothesis-first robustness audits of every fixed parameter (pre-registered in
`sensitivity/HYPOTHESES.md`, embedded per section in RESULTS_REPORT.pdf).
5. **Nuisance parameters verified nuisance** (tested, not assumed): graph vs MC
   seed (eps_c std 0.0025 ≪ grid step, indistinguishable contributions);
   eps-grid step (interpolated estimator off ≤ 0.005 at step 0.05, naive
   convention biased ~ a full step); zealot strategy label (exact R→P→S
   symmetry — permutation test consistent with pure false positives, and
   relabeled runs literally *coalesce* to machine-precision-identical
   trajectories); damage realisation (which edges die shifts eps_c ≤ 0.012 —
   only `2E'/N'` matters, realisation by realisation).
6. **Regime choices hold with margin**: 1500 sweeps is 2x past convergence
   (short runs *under*estimate eps_c — the ordered-side bias wins, not the
   predicted cycling-side one); T=0.65 sits deep in the k-dominated regime
   (eps_c drop over T∈[0.3,1.3]: 0.20 at k=8 → 0.03 at k=40), which is what
   licenses "stability is a function of `<k>` alone".
7. **New physics from the failures — the transition is first-order-like**: the
   HMF has a genuine bistable window (consensus + limit cycle coexisting;
   [0.63,0.71] at k=10, [0.71,0.81] at k=20 — the init picks the attractor),
   and the MC pseudo-transition shifts as 1/N toward eps_c(∞)≈0.58 at k=20 —
   the standard first-order finite-size scaling, found independently in mean
   field and MC. Also: the HMF overestimate of eps_c *reverses* at high k
   (saturated rates plateau near 0.7 while the MC boundary keeps rising), so
   "mean field overestimates order" is a low-k statement.

### The analytic skeleton — report Sec. 0.2 + per-section Mathematics blocks
8. Every report section now carries the governing equations and a derived
   model of its behaviour, each closed form re-checked numerically at report
   build time (the checks live in `build_report.py`). Highlights:
   - **Linearisation**: the HMF Jacobian at the symmetric point reduces on the
     simplex to `1/4·I + k/(4T)·(I + eps·S)`, eigenvalues
     `1/4 + k/4T ± i·sqrt(3)·eps·k/(4T)` — cyclic dominance enters as *pure
     rotation* (verified to 1e-8 against finite differences). The mixed state
     is always unstable; the transition is attractor competition, hence
     first-order-like.
   - **Exact k/T invariance**: the HMF map depends on (k,T) only through k/T,
     so eps_c^HMF = F(k/T) exactly (equal to the last digit at matched (k,T)
     pairs); the quenched MC *breaks* it (0.625 vs 0.698 at k/T=30.8) —
     degree fluctuations sigma_k/<k> = 1/sqrt(<k>) can't be rescaled away.
   - **Bistable window edges located**: Newton continuation puts the true end
     of the ordered branch at eps=0.716 (k=10) / 0.834 (k=20), just above the
     init-measured window tops 0.706/0.806 (basin exit precedes branch death).
   - **Zealot field**: h = kz·(1, eps, −eps) — zealots feed their predator
     and starve its only threat; both free consensuses stay locally stable, so
     zealot outcomes are basin selection (large-z flips onto Rock are ~e^(−cN)
     basin escapes — why the flip fraction dies with N). Two factions:
     h = kz·(1−eps, 1+eps, 0) — "Scissors profits" fails *algebraically*.
   - **Hub leverage**: z_eff = degree-weighted zealot fraction = sqrt(z) on
     ideal BA (measured stub share 0.320 vs sqrt(0.1)=0.316).
   - **Thinning theorem**: edge or node removal maps ER(N,p) to ER with
     <k>_eff = (1−f)<k> — the 6.5 collapse is a closure property of ER, and
     the edge/node coincidence is exact in law, not an empirical accident.
