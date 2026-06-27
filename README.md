# RPS-Potts on Networks — Recreation

A from-scratch rebuild of "Project B" (hybrid Potts + Rock-Paper-Scissors on
complex networks) for learning, built in teaching order: understand the physics
in slow/clear Python, then make it fast, then scale, then iterate.

Original repo: `../RPS_Potts_Network` (thesis Chapter 4, U. Shekhar / C. Hens).

## The model in one paragraph

Each node of a network holds one strategy in {rock, paper, scissors}. Payoffs
come from `P = I + eps*skew`: the identity rewards matching neighbours (drives
ORDER), the skew part is the RPS cycle (drives CYCLING). Nodes update by a
Glauber rule at temperature `T`. The order parameter `m_psi` (magnitude of the
time-averaged complex RPS vector) is ~1 when the network orders and ~0 when it
cycles. Central question: does higher connectivity `<k>` protect order against
the cyclic-dominance strength `eps`?  (Answer: yes.)

## Setup

```bash
python3 -m venv --system-site-packages .venv   # reuse system numpy/matplotlib
./.venv/bin/pip install networkx
```
C++ needs g++ with C++20 (`make -C phase3_cpp`).

## Phases

**Shared code (after refactor):**

| Dir | What it holds |
|-----|---------------|
| `drivers/`  | the single C++ engine `mc_engine` (plain MC + optional zealots) + Makefile + xoshiro.h |
| `common/`   | shared Python: `graphs` (build/write/degree-dist), `observables` (m_psi), `meanfield` (HMF+DMF), `mc_python` (reference MC), `runner` (call the engine) |

**Phases (each imports from `common/`, calls `drivers/mc_engine`):**

| Dir | What it builds | Key output |
|-----|----------------|------------|
| `phase1_hmf/`     | Homogeneous mean field (3 ODEs, no network) | transition curve; order vs cycling |
| `phase2_mc/`      | Agent-level Monte Carlo on ER/BA; MC-vs-HMF validation | finite-size physics the mean field misses |
| `phase3_cpp/`     | Validates the C++ engine vs pure Python (~40x faster) | trust + speed |
| `phase4_pipeline/`| Parallel `(k, eps)` sweep -> phase diagram heatmap | the headline "connectivity stabilises order" figure |
| `phase5_meanfield/`| Degree-based mean field + MC/HMF/DMF comparison | DMF beats HMF, more so on BA |
| `phase6_figures/` | ternary simplex, finite-size scaling, fixed-point stability | the standard figure set |
| `phase7_zealots/` | stubborn-node experiments (random + hub placement) | zealots provoke their own predator |

Run examples:
```bash
./.venv/bin/python phase1_hmf/hmf.py
./.venv/bin/python phase2_mc/mc.py --graph ER --n 500 --k 10
bash phase4_pipeline/run.sh ER          # or BA
./.venv/bin/python phase5_meanfield/compare_suite.py --graph BA --k 10
```

## What's next (iteration ideas, from the thesis future-work)

- **Stubborn nodes / zealots**: a few nodes never change strategy. Can a tiny
  minority flip the whole network?
- **Defects**: quench random nodes/edges (possibly time-varying).
- **Opinion-dynamics coupling**.

## Notes / gotchas

- In the ORIGINAL repo the C++ filenames are swapped: `hmf_simulation.cpp` is
  really DMF, `sde_simulation.cpp` is really HMF. This recreation names things
  correctly.
- `phase4_pipeline/run.sh` uses GNU parallel if present, else `xargs -P`.
- The venv `python` is a symlink: use `$(cd .venv/bin && pwd)/python`, never
  `realpath` (which resolves the symlink out of the venv).
