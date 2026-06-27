"""Call the C++ engine in drivers/ and parse its output.

One place that knows the binary's CLI and output format, so the experiment
scripts don't each re-implement the subprocess plumbing.
"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVERS = os.path.join(ROOT, "drivers")
ENGINE = os.path.join(DRIVERS, "mc_engine")


def ensure_engine():
    """Build drivers/mc_engine if it is missing or out of date."""
    subprocess.run(["make", "-s", "-C", DRIVERS], check=True)
    return ENGINE


def run_engine(edgelist, eps, temp=0.65, sweeps=1500, burn_in=None, seed=1,
               zealot_frac=0.0, zealot_strategy=0, zealot_target="random",
               zealot_frac_b=0.0, zealot_strategy_b=1):
    """Run one simulation; return dict(m_psi, r, p, s, conversion).

    A second (optional) zealot faction is given by zealot_frac_b/zealot_strategy_b.
    """
    if burn_in is None:
        burn_in = int(sweeps * 0.3)
    out = subprocess.run(
        [ENGINE, "--graph", edgelist, "--epsilon", str(eps), "--temp", str(temp),
         "--sweeps", str(sweeps), "--burn-in", str(burn_in), "--seed", str(seed),
         "--zealot-frac", str(zealot_frac),
         "--zealot-strategy", str(zealot_strategy),
         "--zealot-target", zealot_target,
         "--zealot-frac-b", str(zealot_frac_b),
         "--zealot-strategy-b", str(zealot_strategy_b)],
        capture_output=True, text=True, check=True).stdout.split()
    return {"m_psi": float(out[0]), "r": float(out[1]), "p": float(out[2]),
            "s": float(out[3]), "conversion": float(out[4])}


def run_many(jobs, workers=None):
    """Run many simulations in parallel threads (each is a subprocess, so the GIL
    is not a bottleneck). `jobs` is a list of kwarg dicts for run_engine; returns
    the result dicts in the same order."""
    if workers is None:
        workers = os.cpu_count() or 4
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda kw: run_engine(**kw), jobs))
