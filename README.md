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
C++ needs g++ with C++20 (`make -C drivers`).

## Layout

Folders are named by **what they study**. `WALKTHROUGH.md` lists the files in
the order they were built (the learning path), if you want the guided tour.

**Shared code:**

| Dir | What it holds |
|-----|---------------|
| `drivers/`  | the single C++ engine `mc_engine` (plain MC + optional zealots) + Makefile + xoshiro.h |
| `common/`   | shared Python: `graphs` (build/write/degree-dist/defects), `observables` (m_psi), `meanfield` (HMF+DMF), `mc_python` (reference MC), `runner` (call the engine, `run_many`) |

**Studies (each imports from `common/`, calls `drivers/mc_engine`):**

| Dir | What it covers | Key output |
|-----|----------------|------------|
| `mean_field/`   | HMF + DMF analytical models; MC/HMF/DMF comparison | transition curve; DMF beats HMF on BA |
| `monte_carlo/`  | agent-level MC reference; MC-vs-HMF; engine validation | finite-size physics; ~40x speedup check |
| `phase_diagram/`| parallel `(k, eps)` sweep -> phase-diagram heatmap | the "connectivity stabilises order" figure |
| `dynamics/`     | ternary simplex, finite-size scaling, fixed-point stability | the standard figure set |
| `zealots/`      | stubborn-node experiments (random / hub / mixed factions) | zealots provoke their own predator |
| `defects/`      | edge/node quenching (quenched disorder) | defects erode order via effective <k> |

Each section folder has its own `FINDINGS.md` with detailed results; the
top-level `FINDINGS.md` summarises everything (recreation + iteration).

Run examples:
```bash
./.venv/bin/python mean_field/hmf.py
./.venv/bin/python monte_carlo/mc.py --graph ER --n 500 --k 10
bash phase_diagram/run.sh ER          # or BA
./.venv/bin/python mean_field/compare_suite.py --graph BA --k 10
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
- `phase_diagram/run.sh` uses GNU parallel if present, else `xargs -P`.
- The venv `python` is a symlink: use `$(cd .venv/bin && pwd)/python`, never
  `realpath` (which resolves the symlink out of the venv).
