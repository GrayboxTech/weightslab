#!/bin/bash
# Benchmark runner: runs bench.py with W=0, W=2, W=4 back-to-back, with clean state between.
# Writes per-run logs to /tmp/bench_w{N}.log; emits a compact summary at the end.
set -u
cd "$(dirname "$0")"

PY=/home/rotaru/anaconda3/envs/wl_15_nl/bin/python
EPOCHS="${WL_BENCH_EPOCHS:-1}"
WALL_S="${WL_BENCH_WALL_S:-60}"
IMGSZ="${WL_BENCH_IMGSZ:-320}"
LOG_EVERY="${WL_BENCH_LOG_EVERY:-1}"
HARD_TIMEOUT=$(( WALL_S + 90 ))  # headroom for setup/teardown

cleanup() {
  # NOTE: do not pkill -f "bench.py" — that would match the parent bash whose argv contains bench.py
  # Identify bench.py python procs explicitly by matching the python interpreter prefix
  pkill -f "wl_15_nl/bin/python.*bench.py" 2>/dev/null || true
  sleep 1
  for i in $(seq 1 10); do
    if ! ss -lnt 2>/dev/null | grep -q ":50051"; then return 0; fi
    sleep 1
  done
  echo "WARN: port 50051 still in use after 10s" >&2
}

run_one() {
  local W=$1
  local LOG=/tmp/bench_w${W}.log
  echo ""
  echo "========================================================="
  echo "  RUN: workers=$W epochs=$EPOCHS  log=$LOG"
  echo "========================================================="
  cleanup
  WL_BENCH_WORKERS=$W WL_BENCH_EPOCHS=$EPOCHS WL_BENCH_WALL_S=$WALL_S \
    WL_BENCH_IMGSZ=$IMGSZ WL_BENCH_LOG_EVERY=$LOG_EVERY \
    WEIGHTSLAB_LOG_LEVEL=WARNING WEIGHTSLAB_LOG_TO_FILE=false \
    WEIGHTSLAB_SKIP_SECURE_INIT=true \
    timeout -k 5s ${HARD_TIMEOUT}s $PY -u bench.py 2>&1 | tee "$LOG"
  echo "  exit=${PIPESTATUS[0]}"
}

cleanup
for W in 0 2 4; do
  run_one $W
done

echo ""
echo "========================================================="
echo "  SUMMARY"
echo "========================================================="
for W in 0 2 4; do
  LOG=/tmp/bench_w${W}.log
  if [[ -f $LOG ]]; then
    line=$(grep "BENCH END" "$LOG" | tail -1)
    train=$(grep "TRAIN DONE" "$LOG" | tail -1)
    printf "  W=%s  %s\n          %s\n" "$W" "${line:-<no end line — likely timed out>}" "${train:-<no train done>}"
  fi
done
