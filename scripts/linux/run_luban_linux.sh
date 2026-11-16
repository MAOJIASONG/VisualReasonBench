#!/usr/bin/env bash
# Run Puzzle (Unity Luban) experiments on a headless x86_64 Linux machine.
# The harness launches Luban/LuBanSuoEnv.x86_64 itself and talks to it on 127.0.0.1:9999;
# xvfb-run provides the virtual display the Unity player needs to render observations.
#
# Usage (from the project root):
#   bash scripts/linux/run_luban_linux.sh --smoke
#       -> start the Unity player, wait for port 9999, then stop it (no API calls)
#   bash scripts/linux/run_luban_linux.sh --config eval_configs/luban.yaml --levels 0 --runs-per-level 1
#   bash scripts/linux/run_luban_linux.sh --config eval_configs/luban.yaml --level-range 0-31 --workers 8
#       -> any phyvpuzzle.luban_runner arguments are passed through
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
XVFB_ARGS="-screen 0 1280x720x24"

if [ "${1:-}" = "--smoke" ]; then
  echo "Starting Unity player under xvfb..."
  mkdir -p logs
  xvfb-run -a -s "$XVFB_ARGS" ./Luban/LuBanSuoEnv.x86_64 -logFile "$ROOT/logs/unity_smoke.log" &
  PID=$!
  for i in $(seq 1 60); do
    if python - <<'EOF' 2>/dev/null
import socket, sys
s = socket.socket(); s.settimeout(1)
try:
    s.connect(("127.0.0.1", 9999)); sys.exit(0)
except Exception:
    sys.exit(1)
EOF
    then
      echo "OK: Unity server is listening on 127.0.0.1:9999 (after ${i}s)."
      kill "$PID" 2>/dev/null || true
      pkill -f LuBanSuoEnv.x86_64 2>/dev/null || true
      exit 0
    fi
    sleep 1
  done
  echo "FAIL: port 9999 not reachable after 60s. See logs/unity_smoke.log" >&2
  kill "$PID" 2>/dev/null || true
  pkill -f LuBanSuoEnv.x86_64 2>/dev/null || true
  exit 1
fi

# Normal run: the runner (and the Unity player it spawns) share the xvfb display.
exec xvfb-run -a -s "$XVFB_ARGS" python -m phyvpuzzle.luban_runner "$@"
