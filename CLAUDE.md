# CLAUDE.md — RPS-Potts on Networks

Cyclic dominance of a Potts (q=3) + Rock-Paper-Scissors model on complex
networks: MC / HMF / DMF cross-validated, phase diagram, and perturbation
experiments (zealots, hub targeting, competing factions, defects).

Historical note: began as a from-scratch rebuild of thesis "Project B"
(original repo at `../RPS_Potts_Network`), since extended with new analyses.
Folder was renamed `RPS_Potts_Recreation` → `RPS_Potts`; the GitHub remote may
still use the old name. Outward-facing deliverables (RESULTS_REPORT.pdf,
PRESENTING.md) present the work as standalone research — do not reintroduce
"recreation" framing there.

## Model
Nodes play Rock/Paper/Scissors; payoff `P = I + eps*skew` (identity = order,
skew = RPS cycle). Glauber updates at `T=0.65`. Order parameter `m_psi` ~1
ordered/consensus, ~0 cycling. Cycle: Paper>Rock, Scissors>Paper,
Rock>Scissors. eps_c quoted at the (interpolated) `m_psi=0.5` crossing. Key
result: order stability is a function of `<k>` alone (ER≈BA, damaged≈pristine
at matched `<k>`).

## Layout (folders named by topic)
- `drivers/`  — the single C++ engine `mc_engine` (plain MC + optional zealots).
- `common/`   — shared Python: graphs, observables, meanfield (HMF+DMF), mc_python, runner, io.
- `mean_field/`, `monte_carlo/`, `phase_diagram/`, `dynamics/` — core physics.
  Synthesis analyses: `phase_diagram/critical_boundary.py` (eps_c(<k>) ER vs BA
  vs HMF), `defects/collapse.py` (damaged networks land on the pristine boundary).
- `zealots/`, `defects/` — perturbation experiments.
- `sensitivity/` — hypothesis-first robustness audits of every section's fixed
  parameters (T×k, seeds, N, sweeps, eps-grid, MF init, zealot label, damage
  realisation). Protocol: hypotheses pre-registered in
  `sensitivity/HYPOTHESES.md` BEFORE running (method amendments logged there),
  verdicts in `sensitivity/FINDINGS.md`. Key discoveries: the transition is
  first-order-like (HMF bistable window + 1/N shift of MC eps_c), and HMF
  under-predicts eps_c at high k.
- `WALKTHROUGH.md` — build-order learning path. `FINDINGS.md` (root + per folder) — results narrative.

## Reproduction & deliverables
- `./run_all.sh` — the one command: rebuilds the engine, runs the validation
  test, then every simulation step incl. the sensitivity suite (step count in
  `logs/manifest.csv`); per-step logs + `logs/manifest.csv`. Deterministic
  seeds: every figure/CSV must regenerate **byte-for-byte identical** (check
  with `git status` after a run — only `logs/` should change).
- `build_report.py` — generates `RESULTS_REPORT.pdf` (pdflatex). **Every number
  in the report is computed from the CSVs / logs at build time — never
  hardcode results, step counts, dates, or speedups in the LaTeX.** Each result
  section follows the pattern: *What it is* → *Parameters — each defined*
  (gray itemized list: every parameter gets value **and** definition) →
  *How the numbers are obtained* (blue block, references pipeline steps P1–P6
  of report Sec. 0.1) → data table + figure → *Conclusion*. Each section
  additionally ends with *Robustness* subsections (purple *Hypothesis — stated
  before running* block → parameters → data → green *Verdict vs hypothesis*
  block) fed by the `sensitivity/*.csv` tables. All eps_c values use the
  interpolated m_psi=0.5 crossing (same estimator as
  `phase_diagram/critical_boundary.py`, exposed as
  `common.observables.eps_crossing`). Rebuild the report after any data change.
- `RUN_REPORT.md` — written run summary; `PRESENTING.md` — meeting script for
  the professor (what to show, what to say, likely Q&A, the HPC ask).

## Conventions / gotchas
- Python runs in the project venv. Use `$(cd .venv/bin && pwd)/python` — NOT
  `realpath` (it resolves the venv symlink out and loses networkx).
- Build the engine with `make -C drivers` (C++20, needs g++).
- Scripts import shared code via `sys.path.insert(0, <project root>)` then
  `from common import ...`, so they work regardless of folder name.
- `phase_diagram/run.sh` uses GNU parallel if present, else `xargs -P`.
- Every experiment script saves a `.png` figure AND a `.csv` of the underlying
  data next to it; the report reads only the CSVs.
- Engine defaults (via `common/runner.py`): T=0.65, 1500 sweeps, burn-in =
  0.3×sweeps, seed 1.
- Commit messages end with the Co-Authored-By trailer; never commit binaries
  (`drivers/mc_engine`), `.venv`, or `report.tex`/aux files (see `.gitignore`).
