#!/bin/bash
# run_all.sh -- reproduce EVERYTHING from a clean state, with full logging.
#
# Rebuilds the C++ engine, runs the engine-validation test, then executes every
# simulation script from its own folder. Each step is teed to logs/<label>.log,
# and a machine-readable manifest (logs/manifest.csv) records status + timing.
# Regenerates every figure (.png) and data table (.csv) in the repo.
#
# Usage:  ./run_all.sh
set -u
cd "$(dirname "$0")"
ROOT="$(pwd)"
PY="$(cd .venv/bin && pwd)/python"        # project venv (has networkx); abs dir keeps symlink intact
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUNLOG="$LOGDIR/run_${STAMP}.log"
MANIFEST="$LOGDIR/manifest.csv"
echo "label,status,seconds,started_iso,log" > "$MANIFEST"

say() { echo "$@" | tee -a "$RUNLOG"; }

# run <label> <workdir> <cmd...>
run() {
    local label="$1"; shift
    local wd="$1"; shift
    local log="$LOGDIR/${label}.log"
    local ts; ts="$(date -Iseconds)"
    local start; start="$(date +%s)"
    say ""
    say "===== [$label] $* (in $wd) ====="
    ( cd "$wd" && "$@" ) >"$log" 2>&1
    local rc=$?
    local dur=$(( $(date +%s) - start ))
    local status; [ $rc -eq 0 ] && status="OK" || status="FAIL"
    echo "${label},${status},${dur},${ts},logs/${label}.log" >> "$MANIFEST"
    say "  -> ${status} in ${dur}s  (log: logs/${label}.log)"
    # tail a few lines into the combined run log for at-a-glance review
    tail -n 6 "$log" | sed 's/^/    | /' >> "$RUNLOG"
}

say "########## FULL RUN ${STAMP} ##########"
say "host: $(hostname)   cores: $(nproc)   python: $($PY --version 2>&1)"
say "git:  $(git rev-parse --short HEAD)  ($(git rev-parse --abbrev-ref HEAD))"

# --- 0. clean rebuild of the engine ---
say ""; say "===== [engine] clean rebuild ====="
make -s -C drivers clean && make -s -C drivers && say "  -> engine built OK" || say "  -> engine build FAILED"

# --- 1. correctness test: C++ engine vs pure-Python MC ---
run "test_validate_engine"      "monte_carlo"    "$PY" validate_engine.py

# --- 2. mean field (HMF single, HMF sweep, MC/HMF/DMF comparison suite) ---
run "mean_field_hmf"            "mean_field"     "$PY" hmf.py
run "mean_field_sweep"          "mean_field"     "$PY" sweep.py
run "mean_field_compare_suite_ER"  "mean_field"  "$PY" compare_suite.py --graph ER
run "mean_field_compare_suite_BA"  "mean_field"  "$PY" compare_suite.py --graph BA
run "mean_field_compare_grid"   "mean_field"     "$PY" compare_grid.py

# --- 3. monte carlo (single MC run, MC-vs-HMF overlay, overlay grid across T/k/N + BA) ---
run "monte_carlo_mc"            "monte_carlo"    "$PY" mc.py
run "monte_carlo_compare"       "monte_carlo"    "$PY" compare.py
run "monte_carlo_compare_grid"  "monte_carlo"    "$PY" compare_grid.py

# --- 4. phase diagram (parallel <k> x eps sweep), ER and BA, + boundary extraction ---
run "phase_diagram_ER"          "phase_diagram"  bash run.sh ER
run "phase_diagram_BA"          "phase_diagram"  bash run.sh BA
run "phase_diagram_boundary"    "phase_diagram"  "$PY" critical_boundary.py
run "phase_diagram_extra"       "phase_diagram"  "$PY" extra_diagrams.py

# --- 5. dynamics (ternary, finite-size scaling, fixed-point stability) ---
run "dynamics_ternary"          "dynamics"       "$PY" ternary.py
run "dynamics_fss"              "dynamics"       "$PY" fss.py
run "dynamics_stability"        "dynamics"       "$PY" stability.py

# --- 6. zealots (single-faction, hub vs random, competing factions, T/k/N grid, time signals) ---
run "zealots_experiment"        "zealots"        "$PY" experiment.py
run "zealots_experiment_hubs"   "zealots"        "$PY" experiment_hubs.py
run "zealots_experiment_mixed"  "zealots"        "$PY" experiment_mixed.py
run "zealots_experiment_grid"   "zealots"        "$PY" experiment_grid.py
run "zealots_timeseries"        "zealots"        "$PY" timeseries.py

# --- 7. defects (edge/node quenched disorder) + collapse test ---
run "defects_experiment"        "defects"        "$PY" experiment_defects.py
run "defects_collapse"          "defects"        "$PY" collapse.py

# --- 8. sensitivity suite (hypothesis-first robustness audits; see sensitivity/HYPOTHESES.md)
#     sens_defect_seed needs phase_diagram/critical_boundary.csv, so this section
#     stays after the phase-diagram steps.
run "sens_validation"           "sensitivity"    "$PY" sens_validation.py
run "sens_mf_init"              "sensitivity"    "$PY" sens_mf_init.py
run "sens_temperature"          "sensitivity"    "$PY" sens_temperature.py
run "sens_seeds"                "sensitivity"    "$PY" sens_seeds.py
run "sens_grid"                 "sensitivity"    "$PY" sens_grid.py
run "sens_size"                 "sensitivity"    "$PY" sens_size.py
run "sens_equilibration"        "sensitivity"    "$PY" sens_equilibration.py
run "sens_zealot_symmetry"      "sensitivity"    "$PY" sens_zealot_symmetry.py
run "sens_defect_seed"          "sensitivity"    "$PY" sens_defect_seed.py

say ""
say "########## DONE ##########"
say "manifest: logs/manifest.csv"
column -s, -t "$MANIFEST" | sed 's/^/  /' | tee -a "$RUNLOG"

# fail the script if any step failed
if grep -q ",FAIL," "$MANIFEST"; then
    say ""; say "!!! one or more steps FAILED -- see manifest !!!"
    exit 1
fi
say ""; say "All steps OK."
