#!/usr/bin/env bash
# God-script for the WeightsLab-on-ultralytics DDP integration suite. ONE driver,
# several MODES (phases), emits a single report. Runs explicitly LOCALLY (needs a GPU +
# the usecase dataset) — it is NOT a CI unit test. Drives the ws-detection usecase under
# examples/. Runs unattended (ONE sudo prompt, kept alive). Output lands under
# reports/report_<stamp>/ beside this script.
#
# Phases (override with PHASES="..."):
#   info       host/GPU/versions/git snapshot
#   scenarios  full functional suite (ddp_test_suite.py) — pass/fail + per-scn time + MaxRSS
#   ablation   WL internal tax: ulmanual (hand-rolled logger) vs wl (ddp_ablation.py) —
#              per-section time/RSS/IO/bytes + the wl-ulmanual delta
#   profile    py-spy (Python-frame OWNERSHIP: % wall in WL SDK) + perf (native hotspots,
#              perf stat HW counters) + /proc peak RSS/threads, on the wl ablation AND on
#              PROFILE_SCN scenarios. This is the "use py-spy & perf as much as possible" part.
#
# Knobs (all optional):
#   PHASES="info scenarios ablation profile"   BATCH=16 WORKERS=2 CUDA=1
#   ABLATE_STEPS=256   SCN_ONLY=<substr>  SCN_SKIP=a,b   PROFILE_SCN="curate_lifecycle progressive_resample"
#   SAMPLE_DUR=60 SAMPLE_WARM=20   OUT=<dir>
#
# Examples:
#   ./run_ddp_report.sh                              # the works
#   PHASES="ablation profile" ABLATE_STEPS=256 ./run_ddp_report.sh
#   PHASES=profile PROFILE_SCN="curate_lifecycle" ./run_ddp_report.sh
set -uo pipefail                       # NOT -e: every phase runs even if one fails
cd "$(dirname "$0")"

PY=${PY:-/home/rotaru/anaconda3/envs/wl_15_nl/bin/python}
PYSPY=${PYSPY:-/home/rotaru/anaconda3/bin/py-spy}
PERF=${PERF:-/usr/bin/perf}
GTIME=${GTIME:-/usr/bin/time}

PHASES=${PHASES:-"info scenarios ablation profile"}
BATCH=${BATCH:-16}; WORKERS=${WORKERS:-2}; CUDA=${CUDA:-1}
ABLATE_STEPS=${ABLATE_STEPS:-256}
SAMPLE_DUR=${SAMPLE_DUR:-60}; SAMPLE_WARM=${SAMPLE_WARM:-20}
PROFILE_SCN=${PROFILE_SCN:-"curate_lifecycle progressive_resample"}
RD=${OUT:-reports/report_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$RD"; RD=$(cd "$RD" && pwd)
REPORT="$RD/REPORT.md"

# shared env for every child run
export WL_DDP_BATCH=$BATCH WL_DDP_WORKERS=$WORKERS WL_DDP_CUDA=$CUDA
export WEIGHTSLAB_SKIP_SECURE_INIT=true GRPC_TLS_ENABLED=0 WEIGHTSLAB_LOG_LEVEL=ERROR

say(){ echo -e "$*" | tee -a "$REPORT"; }
hr(){ printf '\n%s\n' "================================================================" | tee -a "$REPORT"; }

# ---- sudo: prompt once, keep alive for the whole run (perf+py-spy need it: paranoid=4, ptrace_scope=1)
HAVE_SUDO=1
echo ">> caching sudo (perf + py-spy need root). One prompt now:"
if sudo -v 2>/dev/null; then
  ( while true; do sudo -n true 2>/dev/null; sleep 50; done ) & KEEPALIVE=$!
  trap 'kill $KEEPALIVE 2>/dev/null' EXIT
else
  HAVE_SUDO=0; echo "!! no sudo — 'profile' phase (perf/py-spy) will be SKIPPED"
fi

want(){ [[ " $PHASES " == *" $1 "* ]]; }

say "# WL-on-ultralytics DDP report"
say "_$(date)_  •  batch=$BATCH workers=$WORKERS cuda=$CUDA  •  dir: \`$RD\`"

# ============================================================ INFO
if want info; then
  hr; say "## host / versions"
  say '```'
  { echo "host:   $(uname -srm)  cores=$(nproc)"
    echo "mem:    $(free -h | awk '/Mem:/{print $2" total, "$7" avail"}')"
    echo "gpu:    $($PY - <<'P' 2>/dev/null
import torch
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-only",
      "| torch", torch.__version__)
P
)"
    echo "ultra:  $($PY -c 'import ultralytics;print(ultralytics.__version__)' 2>/dev/null)"
    echo "python: $($PY --version 2>&1)"
    echo "git:    $(git rev-parse --short HEAD 2>/dev/null)  $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
    echo "perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)  ptrace_scope=$(cat /proc/sys/kernel/yama/ptrace_scope)"
  } | tee -a "$REPORT"
  say '```'
fi

# ============================================================ SCENARIOS
if want scenarios; then
  hr; say "## functional suite (ddp_test_suite.py)"
  log="$RD/scenarios.log"
  env WL_DDP_SCN_TIMING=1 \
      ${SCN_ONLY:+WL_DDP_ONLY=$SCN_ONLY} ${SCN_SKIP:+WL_DDP_SKIP=$SCN_SKIP} \
      "$GTIME" -v "$PY" ddp_test_suite.py >"$log" 2>&1
  rc=$?
  say "exit=$rc  (full log: \`scenarios.log\`)"
  say '```'
  grep -E '^  scenario_|RESULT:' "$log" | tee -a "$REPORT"
  grep -E 'took [0-9.]+s' "$log" | sed 's/^/  /' | tail -20 | tee -a "$REPORT"
  grep -E 'Maximum resident|Elapsed \(wall|context switches' "$log" | sed 's/^\s*/  /' | tee -a "$REPORT"
  say '```'
fi

# ============================================================ ABLATION
if want ablation; then
  hr; say "## ablation — WL internal tax vs hand-rolled logging (steps=$ABLATE_STEPS)"
  for M in ulmanual wl; do
    env WL_ABLATE=$M WL_ABLATE_STEPS=$ABLATE_STEPS \
        "$GTIME" -v "$PY" ddp_ablation.py >"$RD/ablation_$M.log" 2>&1
    say "### mode=$M"; say '```'
    sed -n '/^=====/,/^=====/p' "$RD/ablation_$M.log" | tee -a "$REPORT"
    grep -E '^\[mode=' "$RD/ablation_$M.log" | tee -a "$REPORT"
    grep -E 'Maximum resident' "$RD/ablation_$M.log" | sed 's/^\s*/  /' | tee -a "$REPORT"
    say '```'
  done
  # wl - ulmanual per-section delta = WL's internal machinery above hand-rolled
  say "### wl − ulmanual per-section delta (WL internal tax; decode+loss cancel)"; say '```'
  "$PY" - "$RD/ablation_ulmanual.log" "$RD/ablation_wl.log" <<'P' 2>/dev/null | tee -a "$REPORT"
import re,sys
def parse(p):
    d={}
    for ln in open(p):
        m=re.match(r"\s+(\S.*?)\s+([\d.]+) ms/step",ln)
        if m: d[m.group(1).strip()]=float(m.group(2))
    return d
man,wl=parse(sys.argv[1]),parse(sys.argv[2])
for k in wl:
    if k in man: print(f"  {k:18s} {wl[k]-man[k]:+8.1f} ms   (ulmanual {man[k]:6.1f} -> wl {wl[k]:6.1f})")
P
  say '```'
fi

# ============================================================ PROFILE  (py-spy + perf)
# _profile <label> -- <cmd...> : launch cmd, find its mp.spawn rank children, sample
# them for SAMPLE_DUR with py-spy (ownership) AND perf (native), snapshot /proc, kill.
_profile(){
  local label=$1; shift 2          # drop label and the '--'
  [[ $HAVE_SUDO == 1 ]] || { say "  (skipped $label — no sudo)"; return; }
  say "### profile: $label"
  "$@" >"$RD/${label}_workload.log" 2>&1 & local parent=$!
  sleep "$SAMPLE_WARM"
  if ! kill -0 "$parent" 2>/dev/null; then
    say "  workload died in warmup — see ${label}_workload.log"; return; fi
  local kids; kids=$(pgrep -P "$parent" 2>/dev/null | paste -sd, -)
  [[ -z $kids ]] && kids=$parent
  say "  parent=$parent ranks=[$kids]  sampling ${SAMPLE_DUR}s (py-spy + perf concurrently)"

  # py-spy: one recorder per rank (excludes the idle launcher) -> combined folded
  local folded="$RD/${label}.folded"; : >"$folded"
  local sp=()
  for k in ${kids//,/ }; do
    sudo "$PYSPY" record --pid "$k" --rate 200 --duration "$SAMPLE_DUR" \
         --format raw -o "$RD/${label}.$k.folded" >/dev/null 2>>"$RD/${label}_pyspy.err" & sp+=($!)
  done
  # perf: sample + HW-counter the ranks for the same window
  sudo "$PERF" record -o "$RD/${label}.perf.data" -p "$kids" -- sleep "$SAMPLE_DUR" \
       >/dev/null 2>>"$RD/${label}_perf.err" & local pr=$!
  sudo "$PERF" stat -p "$kids" -- sleep "$SAMPLE_DUR" 2>"$RD/${label}.perfstat" & local ps=$!
  wait "${sp[@]}" "$pr" "$ps" 2>/dev/null

  # /proc peak snapshot per rank (before we kill)
  say '```'
  say "  /proc per-rank peak:"
  for k in ${kids//,/ }; do
    [[ -r /proc/$k/status ]] && awk -v k="$k" \
      '/VmHWM|VmRSS|Threads/{printf "    rank-pid %s  %-8s %s %s\n",k,$1,$2,$3}' /proc/$k/status \
      | tee -a "$REPORT"
  done
  kill "$parent" 2>/dev/null; wait "$parent" 2>/dev/null
  for f in "$RD/$label".*.folded; do sudo chmod 644 "$f" 2>/dev/null; cat "$f" >>"$folded"; done

  # py-spy ownership
  say "  --- py-spy ownership (% wall IP in WL SDK; decode/loss/bridge excluded):"
  FOLDED="$folded" "$PY" aggregate_wl_ownership.py | sed 's/^/  /' | tee -a "$REPORT"
  # perf native hotspots + HW counters
  sudo chmod 644 "$RD/${label}.perf.data" 2>/dev/null
  say "  --- perf native hotspots (top symbols — gloo/libtorch/kernel py-spy can't see):"
  sudo "$PERF" report -i "$RD/${label}.perf.data" --stdio -g none --percent-limit 1 2>/dev/null \
    | grep -E '^\s+[0-9]+\.[0-9]+%' | head -20 | sed 's/^/  /' | tee -a "$REPORT"
  say "  --- perf stat (HW counters over the window):"
  grep -E 'instructions|cycles|context-switches|page-faults|cache-misses|insn per|seconds' \
    "$RD/${label}.perfstat" 2>/dev/null | sed 's/^/  /' | tee -a "$REPORT"
  say '```'
}

if want profile; then
  hr; say "## profile (py-spy ownership + perf native)"
  # clean wl ablation target (huge steps so it outlives the sampling window)
  _profile ablation_wl -- env WL_ABLATE=wl WL_ABLATE_STEPS=100000 "$PY" ddp_ablation.py
  # selected real scenarios
  for scn in $PROFILE_SCN; do
    _profile "scn_$scn" -- env WL_DDP_ONLY="$scn" "$PY" ddp_test_suite.py
  done
fi

hr; say "## done"
say "Full report: \`$REPORT\`  •  raw logs/folded/perf.data alongside it in \`$RD\`"
echo ""; echo ">> REPORT written to $REPORT"
