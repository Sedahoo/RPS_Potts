# CLAUDE.md — RPS-Potts on Networks (recreation)

From-scratch rebuild of thesis "Project B" (hybrid Potts + Rock-Paper-Scissors on
complex networks) + novel iteration experiments. Original repo lives at
`../RPS_Potts_Network`.

## Model
Nodes play Rock/Paper/Scissors; payoff `P = I + eps*skew` (identity = order, skew
= RPS cycle). Glauber updates at temperature `T`. Order parameter `m_psi` ~1
ordered, ~0 cycling. Cycle: Paper>Rock, Scissors>Paper, Rock>Scissors. Key
question: does connectivity `<k>` protect order against cyclic strength `eps`?

## Layout (folders named by topic)
- `drivers/`  — the single C++ engine `mc_engine` (plain MC + optional zealots).
- `common/`   — shared Python: graphs, observables, meanfield (HMF+DMF), mc_python, runner, io.
- `mean_field/`, `monte_carlo/`, `phase_diagram/`, `dynamics/` — recreation (reproduces the thesis).
- `zealots/`, `defects/` — novel iteration experiments.
- `WALKTHROUGH.md` — the build-order learning path. `FINDINGS.md` (root + per folder) — results.

## Conventions / gotchas
- Python runs in the project venv. Use `$(cd .venv/bin && pwd)/python` — NOT
  `realpath` (it resolves the venv symlink out and loses networkx).
- Build the engine with `make -C drivers` (C++20, needs g++).
- Scripts import shared code via `sys.path.insert(0, <project root>)` then
  `from common import ...`, so they work regardless of folder name.
- `phase_diagram/run.sh` uses GNU parallel if present, else `xargs -P`.
- Every experiment script saves a `.png` figure AND a `.csv` of the underlying
  data next to it.
- Commit messages end with the Co-Authored-By trailer; never commit binaries
  (`drivers/mc_engine`) or `.venv` (see `.gitignore`).
