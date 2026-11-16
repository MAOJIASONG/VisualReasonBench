#!/usr/bin/env bash
# Start the Unity Luban player under xvfb, replay solver results, stop the player.
#   bash replay_cluster.sh results/Puzzle_0.json results/Puzzle_1.json ...
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
mkdir -p "$ROOT/logs"
# Unity needs an X display. Use xvfb-run if the node has it, otherwise the Xvfb module.
XVFB=""
if command -v xvfb-run >/dev/null; then
  xvfb-run -a -s "-screen 0 1280x720x24" "$ROOT/Luban/LuBanSuoEnv.x86_64" -logFile "$ROOT/logs/unity_replay.log" &
else
  module load Xvfb >/dev/null 2>&1
  DISP=":$((100 + RANDOM % 800))"
  Xvfb "$DISP" -screen 0 1280x720x24 -nolisten tcp >/dev/null 2>&1 &
  XVFB=$!
  sleep 2
  DISPLAY="$DISP" "$ROOT/Luban/LuBanSuoEnv.x86_64" -logFile "$ROOT/logs/unity_replay.log" &
fi
PID=$!
for i in $(seq 1 120); do
  python -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', 9999))" 2>/dev/null && break
  sleep 1
done
echo "unity port check after ${i}s"
if [ "${1:-}" = "--cmd" ]; then shift; (cd "$HERE" && "$@"); else
for r in "$@"; do python "$HERE/replay.py" "$r"; done
fi
kill "$PID" 2>/dev/null; pkill -f LuBanSuoEnv.x86_64 2>/dev/null || true
[ -n "$XVFB" ] && kill "$XVFB" 2>/dev/null || true
