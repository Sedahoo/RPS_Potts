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
