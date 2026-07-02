# Findings — defects/

Novel extension (not in the thesis): quenched disorder. Damage a dense ER graph
(N=1000, <k>=20) by removing a fraction `f` of edges (disabled links) or nodes
(vacancies), then sweep eps.

## Defects erode order via effective <k> (`experiment_defects.py`)

- The transition slides to **lower eps** as `f` grows: f=0 (<k>~20) -> eps_c~0.63;
  f=0.3 (<k>~14) -> ~0.60; f=0.6 (<k>~8) -> ~0.47; f=0.8 (<k>~4) -> ~0.22.
  A damaged, sparser network sustains order only against weaker cyclic pressure.
- **Edge and node defects give nearly identical curves once matched by the
  resulting <k>.** So order-stability depends on the *effective average degree*,
  not on the damage mechanism (which links/nodes were removed).

Takeaway: defects are "connectivity stabilises order" run in reverse — you reach
a sparse graph by breaking a dense one, and the physics only cares about the
endpoint `<k>`.

## The collapse test (`collapse.py`)

- Plotting each damaged network's (resulting <k>, eps_c) point over the
  pristine-ER boundary from `phase_diagram/`: **all eight points (4 edge + 4
  node, even at different N) land on the pristine curve**, and
  max |eps_c(edge) - eps_c(node)| at matched f is only **0.014**.
- This is the strongest form of the finding: a damaged network is
  indistinguishable from a pristine network of the same average degree.
  Data: `defects_collapse.csv`.
