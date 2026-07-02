# Walkthrough — the build order (learning path)

The folders are organised by topic, but the project was built in *teaching
order*: understand the physics in slow/clear Python, then make it fast, then
scale, then iterate. Follow these steps in sequence to retrace that path.

| Step | File | What you learn |
|------|------|----------------|
| 1 | `mean_field/hmf.py` | The dynamics in its purest form — 3 ODEs, no network. Low eps orders, high eps cycles. |
| 2 | `mean_field/sweep.py` | Sweep eps -> the transition curve; higher `<k>` pushes it right (connectivity stabilises order). |
| 3 | `monte_carlo/mc.py` | Real agents on a real network (ER/BA). Stochastic; the network finally matters. |
| 4 | `monte_carlo/compare.py` | MC vs HMF: agree in the bulk, diverge at the transition (finite-size physics). |
| 5 | `drivers/mc_engine.cpp` + `monte_carlo/validate_engine.py` | Port the hot loop to C++ (~40x). Validate it matches the Python MC. |
| 6 | `phase_diagram/run.sh` | The orchestration pattern: generate graphs -> parallel sweep -> plot. Produces the headline `(<k>, eps)` heatmap. |
| 7 | `mean_field/compare_suite.py` | Degree-based mean field (DMF) + the MC/HMF/DMF comparison. Why two mean-field models exist. |
| 8 | `dynamics/ternary.py`, `dynamics/fss.py`, `dynamics/stability.py` | The standard figure set: simplex orbits, finite-size scaling, fixed-point stability. |
| 9 | `zealots/experiment.py`, `experiment_hubs.py`, `experiment_mixed.py` | First new science: stubborn nodes. See `FINDINGS.md`. |
| 10 | `defects/experiment_defects.py` | Quenched disorder: defects erode order via effective `<k>`. See `FINDINGS.md`. |
| 11 | `phase_diagram/critical_boundary.py`, `defects/collapse.py` | Synthesis analyses: extract the eps_c(`<k>`) boundary (ER vs BA vs HMF), and the collapse test showing damaged networks land on the pristine boundary. |

Shared infrastructure used throughout: the C++ engine in `drivers/` and the
Python library in `common/` (graphs, observables, mean-field engines, runner).

The original thesis "Project B" is reproduced by steps 1-8; steps 9-10 are novel
extensions.
