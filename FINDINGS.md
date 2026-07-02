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
- The `(<k>, eps)` heatmap: ordered region in the upper-left, boundary **curving
  up-right** (connectivity protects order), saturating near eps ~ 0.65; sparse
  rows (`<k>`<=4) cannot order at all.
- **ER vs BA nearly identical** — for the MC, average degree matters, not the
  degree-distribution shape. Quantified by `critical_boundary.py`:
  max |eps_c(ER) − eps_c(BA)| = 0.040 over all 20 degrees, with the HMF
  prediction sitting above the MC boundary at every k.

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
