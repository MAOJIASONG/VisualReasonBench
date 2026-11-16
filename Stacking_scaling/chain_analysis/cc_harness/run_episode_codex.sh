#!/usr/bin/env bash
# Run ONE closed-loop episode of Codex CLI (ChatGPT login) on a CHAIN game level
# (FluidNet / Domino / PipeFit, Experiment 6).
#
#   bash run_episode_codex.sh <level_dir> <run_root> <model> [variant] [effort]
#     level_dir : relative to Stacking_scaling, e.g. type_3/puzzles_50/P08_trace_hard_seed14060
#     model     : Codex model id, e.g. gpt-6.1-sol / gpt-6.1-terra / gpt-6-luna
#     effort    : reasoning effort (default high)
#
# Same game, rules, chain-act wrapper and layout as run_episode.sh (Claude Code), with the
# Codex setup of stacking/run_stack_episode_codex.sh: extra features disabled, writes
# sandboxed to the agent folder, one chain-act per command, and every tool call audited
# afterwards (summarize.py).
#
# The game renderer (Chromium via plotly/kaleido) cannot start inside Codex's sandbox, so
# the game is played by bin/chain-serve, which this script runs outside the sandbox; inside,
# chain-act only drops the action into the hidden folder .rpc/ inside the agent folder and
# prints the answer. The session folder stays read-only to the agent. (.rpc/ is inside the
# agent folder so that Codex lists no extra writable folder to the model: with a separate
# folder, gpt-6-luna kept looking for the observation images there.)
#
# Slurm's per-job /tmp breaks Codex's bubblewrap sandbox on the compute nodes, so the
# script re-runs itself in an unprivileged user+mount namespace where /tmp is a plain
# directory under ~/.cache/codex_tmp (see the stacking script for details).
set -euo pipefail

if [ -z "${CODEX_NS_INNER:-}" ]; then
  export CODEX_NS_INNER=1 HOST_UID="$(id -u)" HOST_GID="$(id -g)"
  export NEWTMP="$HOME/.cache/codex_tmp/${SLURM_JOB_ID:-nojob}_$$"
  mkdir -p -m 700 "$NEWTMP"
  exec unshare --user --map-root-user --mount --propagation private bash -c \
    'mount --bind "$NEWTMP" /tmp && exec unshare --user --map-user="$HOST_UID" --map-group="$HOST_GID" bash "$0" "$@"' \
    "$0" "$@"
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
SS="$(cd "$HERE/../.." && pwd)"            # Stacking_scaling
LEVEL="$1"; RUN="$2"; MODEL="$3"; VARIANT="${4:-base}"; EFFORT="${5:-high}"
MAX_ATTEMPTS="${CHAIN_MAX_ATTEMPTS:-6}"
RETRY_WAIT="${CHAIN_RETRY_WAIT:-600}"

game="${LEVEL%%/*}"
name="${game}__$(basename "$LEVEL")"
PRIV="$RUN/private/$name"
AGENT="$RUN/agent/$name"
RPC="$AGENT/.rpc"
mkdir -p "$RUN/private" "$AGENT" "$RPC"
PY="$(command -v python)"

if [ ! -s "$PRIV/.state.json" ]; then
  (cd "$SS" && "$PY" -m chain_analysis.play_cli start "$LEVEL" "$PRIV" >/dev/null 2>&1) || true
  if [ ! -s "$PRIV/.state.json" ]; then
    echo "$name: could not start the game" >&2
    exit 1
  fi
fi
for f in "$PRIV"/obs_*.png; do b=$(basename "$f"); [ -e "$AGENT/$b" ] || cp "$f" "$AGENT/$b"; done
"$PY" "$HERE/prompts.py" --game "$game" --variant "$VARIANT" --agent codex > "$PRIV/prompt.md"

export CHAIN_PRIV="$PRIV" CHAIN_AGENT="$AGENT" CHAIN_SS="$SS" CHAIN_PY="$PY" CHAIN_ONE_PER_CMD=1
export PATH="$HERE/bin:$PATH"

# game server, outside the sandbox; chain-act inside the sandbox talks to it via $RPC
rm -f "$RPC"/req_* "$RPC"/resp_* "$RPC"/done_*
"$HERE/bin/chain-serve" "$RPC" >> "$PRIV/server.log" 2>&1 &
SERVER=$!
trap 'kill "$SERVER" 2>/dev/null || true' EXIT
export CHAIN_RPC_DIR="$RPC"

game_status() { (cd "$SS" && "$PY" -m chain_analysis.play_cli status "$PRIV" 2>/dev/null) \
  | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("status","?"))' 2>/dev/null || echo "?"; }

# thread id / whether the run ended in an error, from a codex --json transcript
tr_field() { "$PY" - "$1" "$2" <<'EOF'
import json, sys
path, key = sys.argv[1], sys.argv[2]
tid, err, done = "", False, False
for line in open(path):
    try:
        ev = json.loads(line)
    except Exception:
        continue
    t = ev.get("type")
    if t == "thread.started":
        tid = ev.get("thread_id", tid)
    elif t in ("turn.failed", "error"):
        err = True
    elif t == "turn.completed":
        done = True
print(tid if key == "thread" else ("True" if err or not done else "False"))
EOF
}

FLAGS=(--json --skip-git-repo-check --ignore-user-config
       -m "$MODEL" -c "model_reasoning_effort=\"$EFFORT\""
       -s workspace-write
       -c 'web_search="disabled"'
       --disable multi_agent --disable multi_agent_v2 --disable browser_use --disable browser_use_external
       --disable in_app_browser --disable computer_use --disable image_generation
       --disable apps --disable plugins --disable goals --disable memories
       --disable skill_search --disable tool_suggest --disable unified_exec)

cd "$AGENT"
thread=""
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  tr="$PRIV/transcript_$attempt.jsonl"
  if [ -z "$thread" ]; then
    codex exec "${FLAGS[@]}" -C "$AGENT" "$(cat "$PRIV/prompt.md")" \
      < /dev/null > "$tr" 2>> "$PRIV/codex_stderr.log" || true
  else
    codex exec resume "$thread" "${FLAGS[@]}" \
      "You were interrupted by an error. Continue the same game from where you stopped: look at the newest obs_*.png images in your working folder and keep playing with chain-act." \
      < /dev/null > "$tr" 2>> "$PRIV/codex_stderr.log" || true
  fi
  t="$(tr_field "$tr" thread)"; [ -n "$t" ] && thread="$t"
  # keep Codex's full session record (every tool call, incl. image views) for the audit
  if [ -n "$thread" ]; then
    ro="$(grep -rl --include="rollout-*.jsonl" "$thread" "${CODEX_HOME:-$HOME/.codex}/sessions" 2>/dev/null | head -1 || true)"
    [ -n "$ro" ] && cp "$ro" "$PRIV/rollout_$attempt.jsonl"
  fi
  is_error="$(tr_field "$tr" error)"
  st="$(game_status)"
  echo "$(date -Is) attempt=$attempt status=$st is_error=$is_error thread=$thread model=$MODEL effort=$EFFORT" >> "$PRIV/episode.log"
  if [ "$st" != "playing" ] || [ "$is_error" != "True" ]; then break; fi
  [ "$attempt" -lt "$MAX_ATTEMPTS" ] && sleep "$RETRY_WAIT"
done
echo "$name status=$(game_status)"
