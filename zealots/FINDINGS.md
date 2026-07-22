# Findings — zealots/

Novel extension (not in the thesis): stubborn nodes locked to a fixed strategy.
Cycle reminder: Paper beats Rock, Scissors beats Paper, Rock beats Scissors.

## 1. Zealots provoke their own predator (`experiment.py`)

A fraction `z` of nodes locked to **Rock**, on ER, averaged over 12 seeds.

- **Ordering phase (eps=0.3):** a few Rock-zealots do NOT make the network adopt
  Rock — the free network flips to **Paper** (which beats Rock). Conversion-to-
  Rock is non-monotonic: ~0 for z in [0.05, 0.16], then partial recovery;
  `m_psi` shows a **frustration minimum** near z ~ 0.16.
- **Cycling phase (eps=0.9):** zealots induce only weak order (`m_psi` ~ linear
  in z, ~0.17 at z=0.2) and cannot pin their own strategy.

Takeaway: the naive "stubborn minority drags everyone along" fails in a cyclic
game — zealots summon their predator.

## 2. Hubs amplify zealots ~8x — but only the *whether* (`experiment_hubs.py`)

Rock-zealots on BA **hubs** vs random nodes (avg of 15 graphs).

- **Cycling phase:** hub placement drives `m_psi` to ~0.72 (linear in z) vs ~0.08
  for random — ~8x amplification. But the order is still **Paper**
  (Rock-conversion -> 0). Hubs control *whether* the network orders, not *what*
  it orders on.
- **Ordering phase:** a crossover — hubs provoke Paper at small z, but **pin
  Rock** (conversion -> 1) once z > 0.08, as the top-degree nodes dominate their
  neighbourhoods.

## 3. Competing factions: the predator wins (`experiment_mixed.py`)

Equal fractions `z` of **Rock** and **Paper** zealots (total 2z), ER.

- **Ordering phase:** at high z the network goes to **Paper**, not Scissors.
  Paper is reinforced twice — by its own zealots and by Rock-zealots provoking
  their predator. Intermediate z is a frustrated, multistable regime.
- **Cycling phase:** robust; balanced zealots barely perturb it (`m_psi` ~ 0.08).

## Across T, <k>, N (`experiment_grid.py`)
Single Rock-faction experiment over T∈{0.4,0.65,1.0}, k∈{6,10,20},
N∈{400,800,1600}: the backfire is generic — small-z zealots elect Paper in
every cell (rho_paper ≈ 0.85–0.95 at z=0.05), never Rock. Hotter systems
re-pin Rock at large z less readily; the cycling phase is immune everywhere;
N is a null axis at small z (curves coincide within seed noise) and merely
bimodal-noisy at large z.

## Time signals (`timeseries.py`)
Per-sweep (r,p,s) via the engine's --timeseries flag (log time axis —
consensus forms in ~10 sweeps on a dense random graph). Clean ordering:
whoever leads the random start snowballs (majority by sweep ~6). Backfire:
zealots make Rock the initial leader (0.35), yet Paper is a majority by sweep
4 and Rock is eaten to its zealot floor. Large faction (z=0.2): Rock first
GROWS to 0.61 feeding on Scissors, then Paper catches up (sweep 8) and pins
it at the floor — the cyclic mechanism in real time. Cycling: no winner ever;
zealots change nothing.

## Time signals for the whole z-sweep (`phase7_timeseries.py`)
Same idea, but every z on the Phase-7 grid (17 points, one ER seed) instead of
4 hand-picked scenarios: per-sweep conversion(t) and the instantaneous
|psi(t)| (not time-averaged), colour-graded by z. Ordering phase: every
trajectory is decided fast (median sweep 6 to move >0.15 off the 1/3
baseline) then flat for the rest of the run; per-realisation the outcome is
**binary** (conversion -> 0 or -> 1), not the smooth partial value the
12-seed average shows — this single seed disagrees with the `experiment.py`
ensemble average by RMSE 0.35 over the z-grid (e.g. z=0.20 crashes to ~0 here
but averages to 0.34 over 12 seeds). That's not a discrepancy to fix: it's
the large-z recovery caught in the act of being a collective-fluctuation /
basin-selection effect (report Sec. 6.1's zealot-field mathematics,
basin escape ~ e^(-cN)) — each realisation lands in one basin, only the
ensemble average looks smooth. Cycling phase: no fast decision anywhere;
conversion and |psi(t)| stay near their baselines and are visibly
z-independent throughout.
