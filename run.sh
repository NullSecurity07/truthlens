#!/usr/bin/env bash
#
# TruthLens launcher.
#
#   ./run.sh                start everything (sets up anything missing first)
#   ./run.sh --setup        first-run setup only: venv, deps, corpus. Does not start.
#   ./run.sh --backend      backend only, no dashboard
#   ./run.sh --local        start with live Wikipedia lookups disabled
#   ./run.sh --stop         stop whatever this script started
#   ./run.sh --status       show what is running
#   ./run.sh --check        run the test suite and the health checks
#   ./run.sh --help
#
# Everything runs on this machine. No API key, no account. The only outbound calls are
# the one-time model download and, unless --local is passed, live Wikipedia lookups.

set -euo pipefail
cd "$(dirname "$0")"
# Absolute, because subshells cd into backend/ and a relative "../.venv/bin/python" makes
# Python emit sys.prefix warnings on every run.
ROOT="$(pwd)"

BACKEND_PORT=8000
DASH_PORT=5173
VENV=".venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
RUNDIR=".run"
BACKEND_LOG="$RUNDIR/backend.log"
DASH_LOG="$RUNDIR/dashboard.log"

# ---------------------------------------------------------------- pretty output
if [ -t 1 ]; then
  B=$'\e[1m'; DIM=$'\e[2m'; GRN=$'\e[32m'; YLW=$'\e[33m'; RED=$'\e[31m'; CYN=$'\e[36m'; R=$'\e[0m'
else
  B=""; DIM=""; GRN=""; YLW=""; RED=""; CYN=""; R=""
fi
step() { printf '%s==>%s %s\n' "$CYN" "$R" "$*"; }
ok()   { printf '  %s✓%s %s\n' "$GRN" "$R" "$*"; }
warn() { printf '  %s!%s %s\n' "$YLW" "$R" "$*"; }
die()  { printf '\n%serror:%s %s\n' "$RED" "$R" "$*" >&2; exit 1; }

port_busy() {
  python3 - "$1" <<'PY' 2>/dev/null
import socket, sys
s = socket.socket()
s.settimeout(0.4)
sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
}

# ---------------------------------------------------------------- preflight
need_tools() {
  command -v python3 >/dev/null || die "python3 not found. Install Python 3.11+ and retry."
  if [ "$WANT_DASHBOARD" = "1" ]; then
    command -v npm >/dev/null || warn "npm not found — the dashboard will be skipped."
  fi
}

setup() {
  mkdir -p "$RUNDIR"

  if [ ! -x "$PY" ]; then
    step "Creating the Python environment (one time)"
    python3 -m venv "$VENV" || die "could not create a virtualenv at $VENV"
    ok "created $VENV"
  fi

  # torch is the marker: if it imports, the backend deps are installed.
  if ! "$PY" -c "import torch, fastapi, transformers, sentence_transformers" >/dev/null 2>&1; then
    step "Installing Python dependencies (one time, a few minutes)"
    "$PIP" install -q --upgrade pip
    # CPU wheels: much smaller than the CUDA build and fast enough for these models.
    "$PIP" install -q --index-url https://download.pytorch.org/whl/cpu torch \
      || die "could not install torch"
    "$PIP" install -q -r backend/requirements.txt || die "could not install backend requirements"
    ok "backend dependencies installed"
  fi

  if [ "$WANT_DASHBOARD" = "1" ] && command -v npm >/dev/null; then
    if [ ! -d dashboard/node_modules ]; then
      step "Installing dashboard dependencies (one time)"
      (cd dashboard && npm install --no-fund --no-audit >/dev/null 2>&1) \
        || warn "npm install failed — the dashboard will be skipped"
      [ -d dashboard/node_modules ] && ok "dashboard dependencies installed"
    fi
  fi

  if [ ! -f backend/data/claims_emb.npy ]; then
    step "Building the fact-check corpus (one time, ~5-10 minutes)"
    echo "     Downloads three public datasets and embeds ~42,000 claims."
    (cd backend && "$ROOT/$PY" scripts/build_corpus.py) || die "corpus build failed"
    ok "corpus built"
  fi
}

# ---------------------------------------------------------------- start / stop
stop_one() {
  # Deliberately two statements: bash expands all words before `local` runs, so
  # `local name="$1" pidfile="$RUNDIR/$name.pid"` would read an unset $name.
  local name="$1"
  local pidfile="$RUNDIR/$name.pid"
  [ -f "$pidfile" ] || return 0
  local pid; pid="$(cat "$pidfile" 2>/dev/null || true)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.3
    done
    kill -9 "$pid" 2>/dev/null || true
    ok "stopped $name (pid $pid)"
  fi
  rm -f "$pidfile"
}

stop_all() {
  mkdir -p "$RUNDIR"
  stop_one backend
  stop_one dashboard
  # anything left holding the ports from an earlier session
  for port in "$BACKEND_PORT" "$DASH_PORT"; do
    if port_busy "$port" && command -v fuser >/dev/null; then
      fuser -k "${port}/tcp" >/dev/null 2>&1 || true
      warn "freed port $port from a process this script did not start"
    fi
  done
}

wait_for_health() {
  local tries="$1" n=0
  while [ "$n" -lt "$tries" ]; do
    if curl -fsS -m 2 "http://127.0.0.1:$BACKEND_PORT/health" >/dev/null 2>&1; then
      return 0
    fi
    # if the backend died, stop waiting and show why
    local started; started="$(cat "$RUNDIR/backend.pid" 2>/dev/null || true)"
    if [ -n "$started" ] && ! kill -0 "$started" 2>/dev/null; then
      return 2
    fi
    n=$((n + 1))
    sleep 1
  done
  return 1
}

start() {
  mkdir -p "$RUNDIR"

  if port_busy "$BACKEND_PORT"; then
    die "port $BACKEND_PORT is already in use. Run './run.sh --stop' first."
  fi

  step "Starting the analysis service on port $BACKEND_PORT"
  ( cd backend && exec "$ROOT/$VENV/bin/uvicorn" app.main:app \
      --host 127.0.0.1 --port "$BACKEND_PORT" ) >"$BACKEND_LOG" 2>&1 &
  OWN_BACKEND_PID=$!
  echo "$OWN_BACKEND_PID" > "$RUNDIR/backend.pid"

  echo "     Loading four models. First run downloads them (~1.6 GB); later runs take ~15s."
  case "$(wait_for_health 300; echo $?)" in
    0) ok "analysis service is up" ;;
    2) echo; tail -n 25 "$BACKEND_LOG" >&2; die "the analysis service exited during startup (log above)" ;;
    *) echo; tail -n 25 "$BACKEND_LOG" >&2; die "the analysis service did not become healthy in time (log above)" ;;
  esac

  if [ "$WANT_DASHBOARD" = "1" ] && [ -d dashboard/node_modules ]; then
    if port_busy "$DASH_PORT"; then
      warn "port $DASH_PORT is busy — skipping the dashboard"
    else
      step "Starting the dashboard on port $DASH_PORT"
      ( cd dashboard && exec npm run dev --silent ) >"$DASH_LOG" 2>&1 &
      OWN_DASH_PID=$!
      echo "$OWN_DASH_PID" > "$RUNDIR/dashboard.pid"
      for _ in $(seq 1 40); do port_busy "$DASH_PORT" && break; sleep 0.25; done
      port_busy "$DASH_PORT" && ok "dashboard is up" || warn "dashboard did not start (see $DASH_LOG)"
    fi
  elif [ "$WANT_DASHBOARD" = "1" ]; then
    warn "dashboard dependencies missing — run './run.sh --setup'"
  fi
}

summary() {
  local health; health="$(curl -fsS -m 3 "http://127.0.0.1:$BACKEND_PORT/health" 2>/dev/null || echo '')"
  local claims live models
  if [ -n "$health" ]; then
    claims="$("$PY" -c "import json,sys;print(f\"{json.loads(sys.argv[1])['corpus_claims']:,}\")" "$health" 2>/dev/null || echo '?')"
    live="$("$PY" -c "import json,sys;print(json.loads(sys.argv[1])['engines'].get('live_research',{}).get('status','?'))" "$health" 2>/dev/null || echo '?')"
    models="$("$PY" -c "
import json,sys
e=json.loads(sys.argv[1])['engines']
names=['toxicity','ai_detection','fact_check']
print(sum(1 for n in names if e.get(n,{}).get('status')=='loaded'), '/', len(names))" "$health" 2>/dev/null || echo '?')"
  fi

  printf '\n%sTruthLens is running%s\n\n' "$B" "$R"
  printf '  %-12s http://localhost:%s\n' "dashboard" "$DASH_PORT"
  printf '  %-12s http://localhost:%s/health\n' "service" "$BACKEND_PORT"
  printf '  %-12s load %s./extension%s unpacked at brave://extensions (Developer mode)\n' \
         "extension" "$DIM" "$R"
  if [ -n "$health" ]; then
    printf '\n  %smodels %s loaded · %s fact-checks indexed · live lookups: %s%s\n' \
           "$DIM" "${models// /}" "${claims}" "${live}" "$R"
  fi
  printf '\n  %sUse it:%s select text on any page, then click the TruthLens icon\n' "$B" "$R"
  printf '          (or right-click → Analyze with TruthLens, or Ctrl+Shift+T)\n'
  printf '\n  %slogs: %s and %s · stop with Ctrl+C or ./run.sh --stop%s\n\n' \
         "$DIM" "$BACKEND_LOG" "$DASH_LOG" "$R"
}

status() {
  if port_busy "$BACKEND_PORT"; then ok "analysis service listening on $BACKEND_PORT"
  else warn "analysis service is not running"; fi
  if port_busy "$DASH_PORT"; then ok "dashboard listening on $DASH_PORT"
  else warn "dashboard is not running"; fi
}

check() {
  [ -x "$PY" ] || die "no environment yet. Run './run.sh --setup' first."
  step "Running the test suite"
  (cd backend && "$ROOT/$PY" -m pytest -q) || die "tests failed"
  if port_busy "$BACKEND_PORT"; then
    step "Checking the running service"
    curl -fsS "http://127.0.0.1:$BACKEND_PORT/health" | "$PY" -m json.tool | head -20
  else
    warn "service not running, skipping the live health check"
  fi
}

# Print the header comment block as the help text, stopping at the first line of code
# rather than at a hardcoded line number that drifts whenever the header is edited.
usage() { awk 'NR>2 && /^#/ { sub(/^# ?/, ""); print; next } NR>2 { exit }' "$0"; }

# ---------------------------------------------------------------- arguments
WANT_DASHBOARD=1
ACTION="start"
for arg in "$@"; do
  case "$arg" in
    --setup)    ACTION="setup" ;;
    --backend)  WANT_DASHBOARD=0 ;;
    --local)    export TRUTHLENS_LIVE_RESEARCH=0 ;;
    --stop)     ACTION="stop" ;;
    --status)   ACTION="status" ;;
    --check)    ACTION="check" ;;
    -h|--help)  usage; exit 0 ;;
    *)          die "unknown option: $arg (try --help)" ;;
  esac
done

case "$ACTION" in
  stop)   stop_all; exit 0 ;;
  status) status; exit 0 ;;
  check)  check; exit 0 ;;
  setup)  need_tools; setup; printf '\n%sSetup complete.%s Start it with ./run.sh\n\n' "$B" "$R"; exit 0 ;;
esac

need_tools
setup

# Ctrl+C should take the whole thing down, not leave orphans behind — but ONLY the
# processes this invocation started. Otherwise refusing to start (port already busy) would
# tear down the instance that is already running, which is the opposite of what is wanted.
OWN_BACKEND_PID=""
OWN_DASH_PID=""
cleanup() {
  [ -n "$OWN_BACKEND_PID$OWN_DASH_PID" ] || return 0
  printf '\n'
  step "Stopping"
  [ -n "$OWN_DASH_PID" ] && stop_one dashboard
  [ -n "$OWN_BACKEND_PID" ] && stop_one backend
  return 0
}
trap cleanup EXIT INT TERM

start
[ "${TRUTHLENS_LIVE_RESEARCH:-1}" = "0" ] && warn "live Wikipedia lookups disabled (--local)"
summary

# Stay in the foreground so Ctrl+C stops both processes.
#
# Two ways this loop ends, and they are not the same thing:
#   - the pid file is gone   -> someone ran './run.sh --stop' from another terminal.
#                               That is deliberate, so exit quietly.
#   - the pid file remains but the process is dead -> it crashed. Say so, and point at
#                               the log rather than exiting silently.
while :; do
  if [ ! -f "$RUNDIR/backend.pid" ]; then
    # Another terminal already stopped these, so the EXIT trap has nothing left to do.
    OWN_BACKEND_PID=""; OWN_DASH_PID=""
    printf '\n  stopped from another terminal.\n\n'
    exit 0
  fi
  # Never fall back to pid 0 here: `kill -0 0` signals this script's own process group
  # and always succeeds, so the loop would spin forever on a dead service.
  watch_pid="$(cat "$RUNDIR/backend.pid" 2>/dev/null || true)"
  if [ -z "$watch_pid" ] || ! kill -0 "$watch_pid" 2>/dev/null; then
    die "the analysis service stopped unexpectedly — see $BACKEND_LOG"
  fi
  sleep 2
done
