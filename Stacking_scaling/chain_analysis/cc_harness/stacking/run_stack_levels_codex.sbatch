#!/usr/bin/env bash
#SBATCH --job-name=stack-codex
#SBATCH --partition=researchshort
#SBATCH --account=eepengresearch
#SBATCH --qos=research-1-qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
#
# Run Codex CLI closed-loop episodes on Stacking levels, for several models one after another.
#   sbatch --chdir=<dir> run_stack_levels_codex.sbatch <first-last> <run_prefix> <model1,model2,..> [variant] [effort]
# Each model writes to <run_prefix>_<model>/. The models run sequentially in one job because
# they share one ChatGPT login (~/.codex/auth.json): parallel jobs refreshing the same token
# could invalidate each other's session.
set -uo pipefail

RANGE="$1"; PREFIX="$2"; MODELS="$3"; VARIANT="${4:-base}"; EFFORT="${5:-high}"
HARNESS="$HOME/programs/2026/CHAIN-linux/Stacking_scaling/chain_analysis/cc_harness/stacking"

source /opt/apps/software/Anaconda3/2024.06-1/etc/profile.d/conda.sh
conda activate "$HOME/envs/phyv_env"
export PATH="$HOME/.local/bin:$PATH"

echo "node=$(hostname) start=$(date -Is) models=$MODELS variant=$VARIANT effort=$EFFORT levels=$RANGE"
for model in ${MODELS//,/ }; do
  RUN="${PREFIX}_${model}"
  mkdir -p "$RUN"
  echo "#### $(date -Is) model $model -> $RUN"
  # Stop the job after 3 episodes in a row fail (e.g. a node or file-system problem), so a
  # broken node cannot burn through the rest of the levels.
  fails=0
  for level in $(seq "${RANGE%-*}" "${RANGE#*-}"); do
    echo "== $(date -Is) $model level $level"
    if bash "$HARNESS/run_stack_episode_codex.sh" "$level" "$RUN" "$model" "$VARIANT" "$EFFORT" < /dev/null; then
      fails=0
    else
      fails=$((fails + 1)); echo "episode failed: $model $level"
      [ "$fails" -ge 3 ] && { echo "3 episodes failed in a row, stopping"; exit 1; }
    fi
  done
  python "$HARNESS/summarize_stack.py" "$RUN"
done
echo "end=$(date -Is)"
