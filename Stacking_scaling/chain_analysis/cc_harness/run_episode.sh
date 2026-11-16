#!/usr/bin/env bash
# Run ONE closed-loop episode of Claude Code on a CHAIN game level.
#
#   bash run_episode.sh <level_dir> <run_root> <model> [variant]
#     level_dir : relative to Stacking_scaling, e.g. type_3/puzzles_50/P08_trace_hard_seed14060
#     run_root  : absolute output folder for this experiment
#     model     : Claude model alias or id, e.g. sonnet / opus
#     variant   : prompt variant, base (default) | plan
#
# Layout:  <run_root>/private/<game>__<level>/  real session (.state.json), prompt, transcripts, logs
#          <run_root>/agent/<game>__<level>/    agent working folder: observation images only
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SS="$(cd "$HERE/../.." && pwd)"            # Stacking_scaling
LEVEL="$1"; RUN="$2"; MODEL="$3"; VARIANT="${4:-base}"
MAX_ATTEMPTS="${CHAIN_MAX_ATTEMPTS:-6}"     # claude launches per episode (resume after API errors)
RETRY_WAIT="${CHAIN_RETRY_WAIT:-600}"       # seconds to wait after an API error / rate limit

game="${LEVEL%%/*}"
name="${game}__$(basename "$LEVEL")"
PRIV="$RUN/private/$name"
AGENT="$RUN/agent/$name"
mkdir -p "$RUN/private" "$AGENT"
PY="$(command -v python)"

if [ ! -f "$PRIV/.state.json" ]; then
  (cd "$SS" && "$PY" -m chain_analysis.play_cli start "$LEVEL" "$PRIV" >/dev/null 2>&1)
fi
for f in "$PRIV"/obs_*.png; do b=$(basename "$f"); [ -e "$AGENT/$b" ] || cp "$f" "$AGENT/$b"; done
"$PY" "$HERE/prompts.py" --game "$game" --variant "$VARIANT" > "$PRIV/prompt.md"

export CHAIN_PRIV="$PRIV" CHAIN_AGENT="$AGENT" CHAIN_SS="$SS" CHAIN_PY="$PY"
export PATH="$HERE/bin:$PATH"
export CLAUDE_CODE_OAUTH_TOKEN="$(cat "$HOME/.claude_oauth_token")"

game_status() { (cd "$SS" && "$PY" -m chain_analysis.play_cli status "$PRIV" 2>/dev/null) \
  | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("status","?"))' 2>/dev/null || echo "?"; }

# Fields of the final "result" event of a stream-json transcript.
result_field() { "$PY" - "$1" "$2" <<'EOF'
import json, sys
path, key = sys.argv[1], sys.argv[2]
res = None
for line in open(path):
    try:
        ev = json.loads(line)
    except Exception:
        continue
    if ev.get("type") == "result":
        res = ev
print("" if res is None else res.get(key, ""))
EOF
}

cd "$AGENT"
session_id=""
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  tr="$PRIV/transcript_$attempt.jsonl"
  args=(-p --model "$MODEL"
        --tools Read,Bash
        --permission-mode dontAsk
        --allowedTools "Read(./**)" "Bash(chain-act:*)"
        --disallowedTools "Read(/$RUN/private/**)" "Read(/$SS/**)" "Read(~/.claude_oauth_token)" "Read(~/.claude/**)"
        --strict-mcp-config
        --output-format stream-json --verbose
        --max-turns 300)
  if [ -z "$session_id" ]; then
    claude "${args[@]}" < "$PRIV/prompt.md" > "$tr" 2>> "$PRIV/claude_stderr.log" || true
  else
    echo "You were interrupted by an API error. Continue the same game from where you stopped: look at the newest obs_*.png images in your working folder and keep playing with chain-act." \
      | claude "${args[@]}" --resume "$session_id" > "$tr" 2>> "$PRIV/claude_stderr.log" || true
  fi
  sid="$(result_field "$tr" session_id)"; [ -n "$sid" ] && session_id="$sid"
  [ -z "$session_id" ] && session_id="$("$PY" -c 'import json,sys
for l in open(sys.argv[1]):
    try: e=json.loads(l)
    except Exception: continue
    if e.get("session_id"): print(e["session_id"]); break' "$tr" 2>/dev/null || true)"
  is_error="$(result_field "$tr" is_error)"
  st="$(game_status)"
  echo "$(date -Is) attempt=$attempt status=$st is_error=$is_error session=$session_id" >> "$PRIV/episode.log"
  # Done unless the agent was cut off by an API error while the game is still running.
  if [ "$st" != "playing" ] || [ "$is_error" != "True" ]; then break; fi
  [ "$attempt" -lt "$MAX_ATTEMPTS" ] && sleep "$RETRY_WAIT"
done
echo "$name status=$(game_status)"
