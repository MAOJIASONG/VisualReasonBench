#!/usr/bin/env bash
# Run ONE closed-loop episode of Codex CLI (ChatGPT login) on a Stacking level.
#
#   bash run_stack_episode_codex.sh <level_index> <run_root> <model> [variant] [effort]
#     model  : Codex model id, e.g. gpt-6.1-sol / gpt-6.1-terra / gpt-6-luna
#     effort : reasoning effort (default high)
#
# Same game, prompt, stack-act wrapper and layout as run_stack_episode.sh (Claude Code).
# Codex cannot allow-list single commands or files, so: every extra feature is disabled
# (sub-agents, web search, browser, plugins, apps, image generation, persistent shell),
# writes are sandboxed to the agent folder + the private session folder, and every
# executed command is audited afterwards (summarize_stack.py).
#
# On the SMU compute nodes Slurm bind-mounts a per-job /tmp from /raid/tmp/<job>/, which
# Codex's bubblewrap sandbox cannot mount ("Can't mkdir parents for /raid/tmp/..."). So the
# script re-runs itself in an unprivileged user+mount namespace where /tmp is a plain
# directory under ~/.cache/codex_tmp, then drops back to the normal uid in a nested user
# namespace. No root or sudo is involved.
set -euo pipefail

# A model's run can be handed to a separate job (to run models in parallel): a file
# <run_root>/.skip_job_<jobid> makes job <jobid> skip every episode of that run.
if [ -n "${SLURM_JOB_ID:-}" ] && [ -e "$2/.skip_job_$SLURM_JOB_ID" ]; then
  echo "skip: $2 is run by another job"
  exit 0
fi

if [ -z "${CODEX_NS_INNER:-}" ]; then
  export CODEX_NS_INNER=1 HOST_UID="$(id -u)" HOST_GID="$(id -g)"
  export NEWTMP="$HOME/.cache/codex_tmp/${SLURM_JOB_ID:-nojob}_$$"
  mkdir -p -m 700 "$NEWTMP"
  exec unshare --user --map-root-user --mount --propagation private bash -c \
    'mount --bind "$NEWTMP" /tmp && exec unshare --user --map-user="$HOST_UID" --map-group="$HOST_GID" bash "$0" "$@"' \
    "$0" "$@"
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
LEVEL="$1"; RUN="$2"; MODEL="$3"; VARIANT="${4:-base}"; EFFORT="${5:-high}"
MAX_ATTEMPTS="${CHAIN_MAX_ATTEMPTS:-6}"
RETRY_WAIT="${CHAIN_RETRY_WAIT:-600}"
PY="$(command -v python)"

lname="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$HERE/levels.json" "$LEVEL")"
name="$(printf 'L%02d_%s' "$LEVEL" "$lname")"
PRIV="$RUN/private/$name"
AGENT="$RUN/agent/$name"
mkdir -p "$RUN/private" "$AGENT"

if [ ! -s "$PRIV/meta.json" ]; then
  "$PY" "$HERE/stack_cli.py" start "$LEVEL" "$PRIV" >/dev/null 2>&1 || true
  if [ ! -s "$PRIV/meta.json" ]; then
    echo "$name: could not start the game" >&2
    exit 1
  fi
fi
for f in "$PRIV"/obs_*.png; do b=$(basename "$f"); [ -e "$AGENT/$b" ] || cp "$f" "$AGENT/$b"; done
"$PY" "$HERE/stack_prompt.py" "$PRIV" --variant "$VARIANT" --agent codex > "$PRIV/prompt.md"

export STACK_PRIV="$PRIV" STACK_AGENT="$AGENT" STACK_PY="$PY"
export PATH="$HERE/bin:$PATH"

game_status() { "$PY" "$HERE/stack_cli.py" status "$PRIV" 2>/dev/null \
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
       -s workspace-write --add-dir "$PRIV"
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
      "You were interrupted by an error. Continue the same puzzle from where you stopped: look at the newest obs_*.png image in your working folder and keep playing with stack-act." \
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
