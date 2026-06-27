# Findings — iteration experiments (phases 7–8)

New results obtained by extending the recreated engine, beyond what the original
thesis reported. Model recap: nodes play Rock/Paper/Scissors, payoff
`P = I + eps*skew`; `eps` is cyclic-dominance strength; `m_psi` ~1 ordered, ~0
cycling. Cycle: Paper beats Rock, Scissors beats Paper, Rock beats Scissors.

## 1. Zealots provoke their own predator  (`phase7_zealots/experiment.py`)

A fraction `z` of nodes locked to **Rock**, on an ER graph, averaged over seeds.

- **Ordering phase (eps=0.3):** a few Rock-zealots do **not** make the network
  adopt Rock — the free network flips to **Paper** (the strategy that beats
  Rock). Conversion-to-Rock is non-monotonic: it drops to ~0 for z in
  [0.05, 0.16], then partially recovers; `m_psi` shows a **frustration minimum**
  near z ~ 0.16 before recovering.
- **Cycling phase (eps=0.9):** zealots induce only weak order (`m_psi` ~ linear
  in z, ~0.17 at z=0.2) and cannot pin their own strategy.

Takeaway: in a cyclic game the naive "stubborn minority drags everyone along"
fails — zealots summon their predator.

## 2. Hubs amplify zealots ~8x — but only the *whether*  (`experiment_hubs.py`)

Same Rock-zealots placed on BA **hubs** vs random nodes (avg of 15 graphs).

- **Cycling phase:** hub placement drives `m_psi` up to ~0.72 (linear in z) vs
  ~0.08 for random — roughly **8x amplification**. But the order is still
  **Paper** (free-node Rock-conversion -> 0). Hubs control *whether* the network
  orders, not *what* it orders on.
- **Ordering phase:** a crossover — hub-zealots provoke Paper at small z, but
  **pin Rock** (conversion -> 1) once z > 0.08, because the top-degree nodes
  directly dominate their neighbourhoods.

## 3. Competing factions: the predator wins  (`experiment_mixed.py`)

Equal fractions `z` of **Rock** and **Paper** zealots (total 2z), ER graph.

- **Ordering phase:** at high z the network goes to **Paper**, not Scissors.
  Paper is reinforced twice — by its own zealots and by Rock-zealots provoking
  their predator (Paper). Intermediate z is a frustrated, multistable regime.
- **Cycling phase:** the cycle is robust; balanced zealots barely perturb it
  (slight Paper tilt, `m_psi` ~ 0.08).

## 4. Defects erode order via effective <k>  (`phase8_defects/experiment_defects.py`)

Quench a fraction `f` of edges or nodes from a dense ER graph (<k>=20), then
sweep eps.

- The transition slides to **lower eps** as `f` grows (f=0 -> eps_c~0.63;
  f=0.8 -> eps_c~0.22): a damaged network sustains order against weaker cyclic
  pressure.
- **Edge and node defects give nearly identical curves when matched by the
  resulting <k>** -> order-stability depends on effective average degree, not on
  the damage mechanism. Defects are "connectivity stabilises order" run in
  reverse.
