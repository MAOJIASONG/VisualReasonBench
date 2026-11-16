#!/usr/bin/env bash
# Run ONE closed-loop episode of Claude Code or Codex CLI on a Puzzle (Luban lock) level.
#
#   bash run_luban_episode.sh <level 0-31> <run_root> <agent claude|codex> <model> [variant] [effort]
#
# Needs a running Unity player (Luban/LuBanSuoEnv.x86_64, see run_luban_lanes.sbatch) and
# LUBAN_ENV_ID (1..8): the Unity environment this episode plays in.
#
# Layout:  <run_root>/private/L<idx>/  game session (meta, prompt, transcripts, logs)
#          <run_root>/agent/L<idx>/    agent working folder: observation images only
#
# Claude Code: same setup as stacking/run_stack_episode.sh (tools Read + Bash, only
# `luban-act` allowed, private folders denied). Codex: same setup as
# stacking/run_stack_episode_codex.sh (extra features disabled, writes sandboxed to the agent
# folder, every tool call audited); its sandbox has no network, so the actions go through
# bin/luban-serve, which runs outside the sandbox (folder .rpc/ inside the agent folder).
set -euo pipefail

AGENT_KIND="$3"
if [ "$AGENT_KIND" = codex ] && [ -z "${CODEX_NS_INNER:-}" ]; then
  # Slurm's per-job /tmp breaks Codex's bubblewrap sandbox: re-run in a user+mount namespace
  # where /tmp is a plain folder (see stacking/run_stack_episode_codex.sh). The network
  # namespace is unchanged, so the Unity socket stays reachable for luban-serve.
  export CODEX_NS_INNER=1 HOST_UID="$(id -u)" HOST_GID="$(id -g)"
  export NEWTMP="$HOME/.cache/codex_tmp/${SLURM_JOB_ID:-nojob}_$$"
  mkdir -p -m 700 "$NEWTMP"
  exec unshare --user --map-root-user --mount --propagation private bash -c \
    'mount --bind "$NEWTMP" /tmp && exec unshare --user --map-user="$HOST_UID" --map-group="$HOST_GID" bash "$0" "$@"' \
    "$0" "$@"
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
LEVEL="$1"; RUN="$2"; MODEL="$4"; VARIANT="${5:-base}"; EFFORT="${6:-high}"
MAX_ATTEMPTS="${CHAIN_MAX_ATTEMPTS:-6}"
RETRY_WAIT="${CHAIN_RETRY_WAIT:-600}"
MAX_ACTIONS="${LUBAN_MAX_ACTIONS:-50}"      # the benchmark's max_steps
: "${LUBAN_ENV_ID:?set LUBAN_ENV_ID (1..8)}"
PY="$(command -v python)"

name="$(printf 'L%02d' "$LEVEL")"
PRIV="$RUN/private/$name"
AGENT="$RUN/agent/$name"
mkdir -p "$RUN/private" "$AGENT"

if [ ! -s "$PRIV/meta.json" ]; then
  mkdir -p "$PRIV"
  "$PY" "$HERE/luban_cli.py" start "$LEVEL" "$PRIV" --env-id "$LUBAN_ENV_ID" --max-actions "$MAX_ACTIONS" \
    > "$PRIV/start.log" 2>&1 || true
  if [ ! -s "$PRIV/meta.json" ]; then
    echo "$name: could not start the game" >&2
    exit 1
  fi
else
  # resuming: point the session at this job's Unity environment (the game itself restarts)
  echo "$name: session exists; not restarting" >&2
fi
for f in "$PRIV"/obs_*.png; do b=$(basename "$f"); [ -e "$AGENT/$b" ] || cp "$f" "$AGENT/$b"; done
"$PY" "$HERE/luban_prompt.py" "$PRIV" --variant "$VARIANT" --agent "$AGENT_KIND" > "$PRIV/prompt.md"

export LUBAN_PRIV="$PRIV" LUBAN_AGENT="$AGENT" LUBAN_PY="$PY"
export PATH="$HERE/bin:$PATH"

game_status() { "$PY" "$HERE/luban_cli.py" status "$PRIV" 2>/dev/null \
  | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("status","?"))' 2>/dev/null || echo "?"; }

# session/thread id and whether the run ended in an error, from a transcript
tr_field() { "$PY" - "$1" "$2" "$AGENT_KIND" <<'EOF'
import json, sys
path, key, kind = sys.argv[1], sys.argv[2], sys.argv[3]
sid, err, done, res = "", False, False, None
for line in open(path):
    try:
        ev = json.loads(line)
    except Exception:
        continue
    t = ev.get("type")
    if kind == "claude":
        sid = sid or ev.get("session_id", "")
        if t == "result":
            res = ev
    else:
        if t == "thread.started":
            sid = ev.get("thread_id", sid)
        elif t in ("turn.failed", "error"):
            err = True
        elif t == "turn.completed":
            done = True
if kind == "claude":
    if res is not None:
        sid = res.get("session_id") or sid
    err = res is None or bool(res.get("is_error"))
else:
    err = err or not done
print(sid if key == "id" else ("True" if err else "False"))
EOF
}

RESUME_MSG="You were interrupted by an error. Continue the same puzzle from where you stopped: look at the newest obs_*.png image in your working folder and keep playing with luban-act."

if [ "$AGENT_KIND" = claude ]; then
  export CLAUDE_CODE_OAUTH_TOKEN="$(cat "$HOME/.claude_oauth_token")"
  ARGS=(-p --model "$MODEL"
        --tools Read,Bash
        --permission-mode dontAsk
        --allowedTools "Read(./**)" "Bash(luban-act:*)"
        --disallowedTools "Read(/$RUN/private/**)" "Read(/$HERE/**)" "Read(~/.claude_oauth_token)" "Read(~/.claude/**)"
        --strict-mcp-config
        --output-format stream-json --verbose
        --max-turns 400)
else
  RPC="$AGENT/.rpc"
  mkdir -p "$RPC"; rm -f "$RPC"/req_* "$RPC"/resp_* "$RPC"/done_*
  "$HERE/bin/luban-serve" "$RPC" >> "$PRIV/server.log" 2>&1 &
  SERVER=$!
  trap 'kill "$SERVER" 2>/dev/null || true' EXIT
  export LUBAN_RPC_DIR="$RPC"
  ARGS=(--json --skip-git-repo-check --ignore-user-config
        -m "$MODEL" -c "model_reasoning_effort=\"$EFFORT\""
        -s workspace-write
        -c 'web_search="disabled"'
        --disable multi_agent --disable multi_agent_v2 --disable browser_use --disable browser_use_external
        --disable in_app_browser --disable computer_use --disable image_generation
        --disable apps --disable plugins --disable goals --disable memories
        --disable skill_search --disable tool_suggest --disable unified_exec)
fi

cd "$AGENT"
sid=""
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  tr="$PRIV/transcript_$attempt.jsonl"
  if [ "$AGENT_KIND" = claude ]; then
    if [ -z "$sid" ]; then
      claude "${ARGS[@]}" < "$PRIV/prompt.md" > "$tr" 2>> "$PRIV/agent_stderr.log" || true
    else
      echo "$RESUME_MSG" | claude "${ARGS[@]}" --resume "$sid" > "$tr" 2>> "$PRIV/agent_stderr.log" || true
    fi
  else
    if [ -z "$sid" ]; then
      codex exec "${ARGS[@]}" -C "$AGENT" "$(cat "$PRIV/prompt.md")" < /dev/null > "$tr" 2>> "$PRIV/agent_stderr.log" || true
    else
      codex exec resume "$sid" "${ARGS[@]}" "$RESUME_MSG" < /dev/null > "$tr" 2>> "$PRIV/agent_stderr.log" || true
    fi
  fi
  s="$(tr_field "$tr" id)"; [ -n "$s" ] && sid="$s"
  if [ "$AGENT_KIND" = codex ] && [ -n "$sid" ]; then
    # keep Codex's full session record (every tool call, incl. image views) for the audit
    ro="$(grep -rl --include="rollout-*.jsonl" "$sid" "${CODEX_HOME:-$HOME/.codex}/sessions" 2>/dev/null | head -1 || true)"
    [ -n "$ro" ] && cp "$ro" "$PRIV/rollout_$attempt.jsonl"
  fi
  is_error="$(tr_field "$tr" error)"
  st="$(game_status)"
  echo "$(date -Is) attempt=$attempt status=$st is_error=$is_error session=$sid agent=$AGENT_KIND model=$MODEL" >> "$PRIV/episode.log"
  # Done unless the agent was cut off by an API error while the game is still running.
  if [ "$st" != "playing" ] || [ "$is_error" != "True" ]; then break; fi
  [ "$attempt" -lt "$MAX_ATTEMPTS" ] && sleep "$RETRY_WAIT"
done
echo "$name status=$(game_status)"
