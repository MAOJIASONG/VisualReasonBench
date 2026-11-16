#!/usr/bin/env bash
#SBATCH --job-name=luban-agents
#SBATCH --partition=researchshort
#SBATCH --account=eepengresearch
#SBATCH --qos=research-1-qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
#
# Puzzle (Luban) closed-loop runs: one Unity player for the whole job, several agents/models in
# parallel, each in its own Unity environment (env_id 1..8) and playing the levels in order.
#
#   sbatch --chdir=<runs dir> run_luban_lanes.sbatch <levels first-last> <run_prefix> <variant> <lane> [<lane> ...]
#     lane = agent:model[:effort], e.g. claude:sonnet claude:opus codex:gpt-6.1-sol:high
#   Each lane writes to <run_prefix>_<agent>_<model>/.
set -uo pipefail

RANGE="$1"; PREFIX="$2"; VARIANT="$3"; shift 3
case "$PREFIX" in /*) ;; *) PREFIX="$PWD/$PREFIX" ;; esac   # episodes cd into their own folders
HARNESS="$HOME/programs/2026/CHAIN-linux/Stacking_scaling/chain_analysis/cc_harness/luban"
ROOT="$HOME/programs/2026/CHAIN-linux"

source /opt/apps/software/Anaconda3/2024.06-1/etc/profile.d/conda.sh
conda activate "$HOME/envs/phyv_env"
export PATH="$HOME/.local/bin:$PATH"

echo "node=$(hostname) start=$(date -Is) levels=$RANGE variant=$VARIANT lanes=$*"

# The Unity player always listens on port 9999, so only one such job can run per node.
if python -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', 9999))" 2>/dev/null; then
  echo "port 9999 is already in use on $(hostname) (another Unity job?); stopping"; exit 1
fi

# Unity needs an X display: the Xvfb module (software rendering, no GPU needed).
module load Xvfb >/dev/null 2>&1
DISP=":$((100 + RANDOM % 800))"
Xvfb "$DISP" -screen 0 1280x720x24 -nolisten tcp >/dev/null 2>&1 &
XVFB=$!
sleep 2
LOG="$PWD/unity-$SLURM_JOB_ID.log"
DISPLAY="$DISP" "$ROOT/Luban/LuBanSuoEnv.x86_64" -logFile "$LOG" &
UNITY=$!
trap 'kill $UNITY $XVFB 2>/dev/null; pkill -P $$ 2>/dev/null' EXIT
for i in $(seq 1 120); do
  python -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', 9999))" 2>/dev/null && break
  sleep 1
done
echo "unity ready after ${i}s"

lane() {   # lane <env_id> <agent> <model> <effort>
  local env="$1" agent="$2" model="$3" effort="$4"
  local run="${PREFIX}_${agent}_${model}"
  mkdir -p "$run"
  local fails=0
  for level in $(seq "${RANGE%-*}" "${RANGE#*-}"); do
    echo "== $(date -Is) $agent $model level $level"
    if LUBAN_ENV_ID="$env" bash "$HARNESS/run_luban_episode.sh" "$level" "$run" "$agent" "$model" "$VARIANT" "$effort" < /dev/null; then
      fails=0
    else
      fails=$((fails + 1)); echo "episode failed: $agent $model $level"
      [ "$fails" -ge 3 ] && { echo "3 episodes failed in a row, stopping lane $agent $model"; break; }
    fi
  done
  python "$HARNESS/summarize_luban.py" "$run"
}

env=1; pids=()
for spec in "$@"; do
  IFS=: read -r agent model effort <<< "$spec"
  lane "$env" "$agent" "$model" "${effort:-high}" > "lane-$SLURM_JOB_ID-$agent-$model.log" 2>&1 &
  pids+=($!)
  env=$((env + 1))
done
wait "${pids[@]}"
echo "end=$(date -Is)"
