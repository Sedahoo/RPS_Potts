#!/bin/bash
# Phase 4, step 2: sweep the (avg-degree, epsilon) grid in parallel and plot the
# phase diagram. This is the orchestration pattern the whole original repo uses:
#   generate inputs  ->  fan a fast C++ binary across cores  ->  plot.
#
# Uses GNU parallel if installed, otherwise falls back to `xargs -P` (built in).
set -e
cd "$(dirname "$0")"

# --- CONFIG (kept modest so it finishes in ~1-2 min on a laptop) ---
GRAPH_TYPE="${1:-ER}"          # ER or BA  (pass as first arg)
N_NODES=800
TEMPERATURE=0.65
SWEEPS=800
BURN_IN=240
EPS_MIN=0.0; EPS_MAX=1.0; EPS_NUM=26
DEG_MIN=2;  DEG_MAX=80;  DEG_STEP=2
NUM_JOBS="$(nproc)"

PYTHON="$(cd ../.venv/bin && pwd)/python"   # project venv (has networkx); abs dir, keeps the venv symlink intact
BIN="../drivers/mc_engine"
GRAPHS_DIR="temp_graphs_${GRAPH_TYPE}"
RESULTS_DIR="temp_results_${GRAPH_TYPE}"
JOBS_FILE="jobs_${GRAPH_TYPE}.txt"
FINAL_PLOT="phase_diagram_${GRAPH_TYPE}.png"

echo "===== Phase 4 pipeline: ${GRAPH_TYPE}, N=${N_NODES}, ${NUM_JOBS} jobs ====="

# --- Step 0: make sure the engine is built ---
make -s -C ../drivers

# --- Step 1: generate one graph per degree ---
echo "--- generating graphs (k=${DEG_MIN}..${DEG_MAX} step ${DEG_STEP}) ---"
mkdir -p "$GRAPHS_DIR" "$RESULTS_DIR"
DEGREES=$(seq "$DEG_MIN" "$DEG_STEP" "$DEG_MAX")
for k in $DEGREES; do
    $PYTHON generate_graphs.py --n "$N_NODES" --avg-degree "$k" \
        --type "$GRAPH_TYPE" --output-dir "$GRAPHS_DIR" --seed 1
done

# --- Step 2: build the list of simulation commands ---
echo "--- building job list ---"
> "$JOBS_FILE"
for k in $DEGREES; do
    GRAPH="${GRAPHS_DIR}/graph_N${N_NODES}_k${k}.edgelist"
    for ((j=0; j<EPS_NUM; j++)); do
        eps=$(awk "BEGIN{printf \"%.5f\", $EPS_MIN + $j*($EPS_MAX-$EPS_MIN)/($EPS_NUM-1)}")
        out="${RESULTS_DIR}/result_k${k}_e${j}.txt"
        echo "$BIN --graph $GRAPH --epsilon $eps --temp $TEMPERATURE --sweeps $SWEEPS --burn-in $BURN_IN --seed 1 --output $out" >> "$JOBS_FILE"
    done
done
NJOBS=$(wc -l < "$JOBS_FILE")
echo "--- running $NJOBS simulations across $NUM_JOBS cores ---"

# --- Step 3: run them in parallel (GNU parallel, else xargs -P) ---
if command -v parallel >/dev/null 2>&1; then
    parallel -j "$NUM_JOBS" --bar < "$JOBS_FILE"
else
    echo "(GNU parallel not found -> using xargs -P fallback)"
    xargs -P "$NUM_JOBS" -d '\n' -I CMD bash -c 'CMD' < "$JOBS_FILE"
fi

# --- Step 4: plot the heatmap ---
echo "--- plotting ---"
DEG_LIST=$(echo $DEGREES | tr ' ' ',')
$PYTHON plot_phase_diagram.py --results-dir "$RESULTS_DIR" --output "$FINAL_PLOT" \
    --degs "$DEG_LIST" --eps-params "$EPS_MIN" "$EPS_MAX" "$EPS_NUM" \
    --graph-type "$GRAPH_TYPE" --n "$N_NODES" --temp "$TEMPERATURE"

# --- Step 5: clean up intermediates ---
rm -rf "$GRAPHS_DIR" "$RESULTS_DIR" "$JOBS_FILE"
echo "Done -> $FINAL_PLOT"
