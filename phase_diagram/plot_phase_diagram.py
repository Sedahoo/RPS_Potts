"""
Phase 4, step 3: assemble the (avg-degree x epsilon) grid of m_psi values into
the phase-diagram heatmap -- the project's headline figure.

Red  (m_psi ~ 1) = ordered phase.
Blue (m_psi ~ 0) = cycling / disordered phase.
The curving boundary between them is the result: as <k> grows (up the y-axis)
the red region extends to higher epsilon -> connectivity stabilises order.
"""

import argparse, os
import numpy as np
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--degs", required=True, help="comma-separated degrees")
    ap.add_argument("--eps-params", nargs=3, type=float, required=True,
                    metavar=("MIN", "MAX", "NUM"))
    ap.add_argument("--graph-type", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--temp", type=float, required=True)
    a = ap.parse_args()

    degs = [int(x) for x in a.degs.split(",")]
    eps_min, eps_max, eps_num = a.eps_params[0], a.eps_params[1], int(a.eps_params[2])
    eps_vals = np.linspace(eps_min, eps_max, eps_num)

    # grid[i, j] = m_psi at degree degs[i], epsilon index j
    grid = np.full((len(degs), eps_num), np.nan)
    for i, k in enumerate(degs):
        for j in range(eps_num):
            path = os.path.join(a.results_dir, f"result_k{k}_e{j}.txt")
            try:
                with open(path) as f:
                    grid[i, j] = float(f.read().split()[0])   # m_psi
            except (FileNotFoundError, IndexError, ValueError):
                pass

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="RdBu_r",
                   vmin=0, vmax=1,
                   extent=[eps_min, eps_max, degs[0], degs[-1]])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$m_\psi$  (1 = ordered, 0 = cycling)")
    ax.set_xlabel(r"$\epsilon$  (cyclic-dominance strength)")
    ax.set_ylabel(r"average degree  $\langle k \rangle$")
    ax.set_title(f"Phase diagram ({a.graph_type}, N={a.n}, T={a.temp})\n"
                 f"higher connectivity protects the ordered phase")
    fig.tight_layout()
    fig.savefig(a.output, dpi=130)
    print(f"Saved {a.output}")

    # save the grid behind the heatmap as long-format CSV (degree, epsilon, m_psi)
    rows = [(k, eps_vals[j], grid[i, j])
            for i, k in enumerate(degs) for j in range(eps_num)]
    csv_path = a.output.rsplit(".", 1)[0] + ".csv"
    np.savetxt(csv_path, np.array(rows), delimiter=",",
               header="degree,epsilon,m_psi", comments="")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
