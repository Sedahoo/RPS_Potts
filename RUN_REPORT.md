# Run Report — full reproduction for presentation

**Date:** 2026-07-02   **Host:** 16 cores   **Engine:** `drivers/mc_engine` (C++20, `-O3 -march=native`), clean-rebuilt
**Driver:** `./run_all.sh` (rebuilds engine → runs validation test → runs every simulation from its folder)
**Logs:** `logs/` (one `<label>.log` per step, machine-readable `logs/manifest.csv`, combined `logs/run_*.log`)

Reproduce everything in ~2 min:

```bash
./run_all.sh          # writes logs/manifest.csv + logs/*.log, regenerates every .png and .csv
```

---

## Headline: everything ran green and is bit-for-bit reproducible

- **18 / 18 steps OK**, exit code 0 (see `logs/manifest.csv`). The two newest
  steps are the synthesis analyses: `phase_diagram/critical_boundary.py`
  (extracted ε_c(⟨k⟩) boundary, ER vs BA vs HMF; max ER−BA gap **0.040**) and
  `defects/collapse.py` (damaged networks collapse onto the pristine boundary;
  edge−node gap **0.014**).
- After the run, `git status` reports **zero** changes to any tracked figure or data table — every `.png` and `.csv` regenerated **byte-for-byte identical** to the committed version. Deterministic seeds → full reproducibility.
- The C++ engine was validated against the independent pure-Python reference before any science was run.

---

## 1. Correctness test — C++ engine vs pure-Python MC

`monte_carlo/validate_engine.py` runs the same graph through both engines (different RNG streams, so agreement is in phase/regime, not last digit).

| epsilon | Python m_ψ | C++ m_ψ | verdict |
|--------:|-----------:|--------:|:-------:|
| 0.2 | 0.999 | 0.999 | agree |
| 0.5 | 0.991 | 0.992 | agree |
| 0.7 | 0.004 | 0.007 | agree |
| 0.9 | 0.004 | 0.003 | agree |

**Speedup ≈ 20×** (Python 3.47 s vs C++ 0.17 s for the 4-point sweep). The engine reproduces the ordered phase below the transition and the cycling phase above it — the physics ground truth is trustworthy.

---

## 2. Results by section (all numbers from this run's regenerated CSVs)

### Core validation (Monte Carlo / mean-field / phase diagram)

**Mean field — `mean_field/`**
- HMF single trajectory + ε-sweep across ⟨k⟩ ∈ {2,5,10,50,200}: connectivity stabilises order (transition moves to higher ε as ⟨k⟩ grows). → `hmf_sweep.csv`
- MC vs HMF vs DMF comparison suite, RMSE against MC ground truth:

  | graph | RMSE(HMF) | RMSE(DMF) | winner | DMF advantage |
  |:-----:|----------:|----------:|:------:|--------------:|
  | ER | 0.3376 | 0.3340 | **DMF** | 0.0036 |
  | BA | 0.3357 | 0.3280 | **DMF** | 0.0077 |

  DMF beats HMF on both, and its edge is **~2× larger on heterogeneous BA** — degree-resolved mean field pays off exactly where the degree distribution is broad. Both mean fields overestimate the ordered phase (MC transition ≈ 0.5 vs mean-field ≈ 0.62); error concentrates in the transition band.

**Monte Carlo — `monte_carlo/`**
- Single MC run on ER + MC-vs-HMF overlay (78 s sweep): agree in bulk, MC transition earlier than HMF → mean field overestimates order. → `mc_vs_hmf.csv`

**Phase diagram — `phase_diagram/`**
- (⟨k⟩ × ε) heatmaps for ER and BA, **520 simulations each**, fanned across 16 cores in ~6 s. Reproduces the "connectivity vs stability" phase boundary; ER ≈ BA (average degree matters more than P(k) shape for MC). → `phase_diagram_{ER,BA}.csv`

**Dynamics — `dynamics/`**
- Ternary (corner attractor vs limit cycle), fixed-point stability (consensus vs limit cycle), and finite-size scaling N ∈ {200, 500, 1000, 2000}: the transition sharpens and converges toward the true critical point as N grows. → `fss.csv`

### Extensions (novel perturbation experiments)

**Zealots — `zealots/`**
- **Single Rock-faction (ordering, ε=0.3):** a few Rock-zealots *provoke their own predator* — free-node conversion-to-Rock collapses from 0.58 → **0.00 by z≈0.05** (the free network flips to **Paper**, the strategy that beats Rock). Counterintuitive: naive minority takeover fails in a cyclic system.
- **Cycling (ε=0.9):** zealots induce only weak order (m_ψ ≈ 0.18 at z=0.20) and cannot pin their strategy.
- **Hub vs random placement (cycling, z=0.10):** hub-placed zealots give m_ψ = **0.72** vs random **0.08** — a **~8.9× amplification**. Hubs control *whether* the network orders, not *what* it orders on. → `zealots_hubs.csv`
- **Competing Rock + Paper factions (ordering, high z):** the free population goes to **Paper (0.90)**, not Scissors (0.00) — the predator reinforced by its own zealots *and* by the Rock-zealots' provocation wins. → `zealots_mixed.csv`

**Defects — `defects/`**
- Quenched disorder (remove a fraction f of edges or nodes) slides the transition to lower ε, tracking the *resulting* mean degree:

  | f (edge defects) | ε_c | resulting ⟨k⟩ |
  |:----:|:---:|:---:|
  | 0.0 | 0.64 | 20.0 |
  | 0.3 | 0.60 | 14.0 |
  | 0.6 | 0.48 | 8.0 |
  | 0.8 | 0.24 | 4.1 |

  Edge and node defects coincide when matched by the resulting ⟨k⟩ → order-stability depends on effective ⟨k⟩ only. → `defects.csv`

---

## 3. Full step manifest

Every step, its wall-clock time and log, is in `logs/manifest.csv`:

| step | status | seconds |
|------|:------:|--------:|
| test_validate_engine | OK | 4 |
| mean_field_hmf | OK | 1 |
| mean_field_sweep | OK | 6 |
| mean_field_compare_suite_ER | OK | 6 |
| mean_field_compare_suite_BA | OK | 6 |
| monte_carlo_mc | OK | 3 |
| monte_carlo_compare | OK | 79 |
| phase_diagram_ER | OK | 6 |
| phase_diagram_BA | OK | 6 |
| phase_diagram_boundary | OK | 0 |
| dynamics_ternary | OK | 2 |
| dynamics_fss | OK | 3 |
| dynamics_stability | OK | 0 |
| zealots_experiment | OK | 3 |
| zealots_experiment_hubs | OK | 4 |
| zealots_experiment_mixed | OK | 2 |
| defects_experiment | OK | 7 |
| defects_collapse | OK | 1 |

Total wall-clock ≈ 2 min (the 79 s `monte_carlo_compare` is the pure-Python HMF overlay; everything else is C++-backed).

---

*Generated artifacts: 12 figures (`*.png`) + 11 data tables (`*.csv`), one per figure, all regenerated by this run. See `FINDINGS.md` for the full physics narrative and each section's `FINDINGS.md` for detail.*
