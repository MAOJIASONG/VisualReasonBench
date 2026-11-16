#!/usr/bin/env bash
#SBATCH --job-name=chain-codex
#SBATCH --partition=researchshort
#SBATCH --account=eepengresearch
#SBATCH --qos=research-1-qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
#
# Run Codex CLI closed-loop episodes on a list of CHAIN levels, for several models one after
# another (they share one ChatGPT login, so they are not run in parallel).
#   sbatch run_levels_codex.sbatch <levels.txt> <run_prefix> <model1,model2,..> [variant] [effort]
# Each model writes to <run_prefix>_<model>/.
set -uo pipefail

LEVELS="$1"; PREFIX="$2"; MODELS="$3"; VARIANT="${4:-base}"; EFFORT="${5:-high}"
HARNESS="$HOME/programs/2026/CHAIN-linux/Stacking_scaling/chain_analysis/cc_harness"

source /opt/apps/software/Anaconda3/2024.06-1/etc/profile.d/conda.sh
conda activate "$HOME/envs/phyv_env"
export PATH="$HOME/.local/bin:$PATH"

echo "node=$(hostname) start=$(date -Is) models=$MODELS variant=$VARIANT effort=$EFFORT levels=$LEVELS"
for model in ${MODELS//,/ }; do
  RUN="${PREFIX}_${model}"
  mkdir -p "$RUN"
  echo "#### $(date -Is) model $model -> $RUN"
  # Stop the job after 3 episodes in a row fail (e.g. a node or file-system problem), so a
  # broken node cannot burn through the rest of the level list.
  fails=0
  while read -r level; do
    echo "== $(date -Is) $model $level"
    if bash "$HARNESS/run_episode_codex.sh" "$level" "$RUN" "$model" "$VARIANT" "$EFFORT" < /dev/null; then
      fails=0
    else
      fails=$((fails + 1)); echo "episode failed: $model $level"
      [ "$fails" -ge 3 ] && { echo "3 episodes failed in a row, stopping"; exit 1; }
    fi
  done < <(grep -v '^\s*#' "$LEVELS" | grep -v '^\s*$')
  python "$HARNESS/summarize.py" "$RUN" --model "codex-$model-$VARIANT" --export
done
echo "end=$(date -Is)"
