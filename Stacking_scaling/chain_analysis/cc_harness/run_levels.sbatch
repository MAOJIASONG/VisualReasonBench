#!/usr/bin/env bash
#SBATCH --job-name=chain-cc
#SBATCH --partition=researchshort
#SBATCH --account=eepengresearch
#SBATCH --qos=research-1-qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out
#
# Run Claude Code closed-loop episodes on a list of CHAIN levels, one after another.
#   sbatch --chdir=<run_root> run_levels.sbatch <levels.txt> <run_root> <model> [variant]
# levels.txt: one level dir per line, relative to Stacking_scaling (lines starting with # ignored).
# No GPU is requested: the agent runs remotely, the node only renders images.
set -uo pipefail

LEVELS="$1"; RUN="$2"; MODEL="$3"; VARIANT="${4:-base}"
HARNESS="$HOME/programs/2026/CHAIN-linux/Stacking_scaling/chain_analysis/cc_harness"

source /opt/apps/software/Anaconda3/2024.06-1/etc/profile.d/conda.sh
conda activate "$HOME/envs/phyv_env"
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$RUN"
echo "node=$(hostname) start=$(date -Is) model=$MODEL variant=$VARIANT"
grep -v '^\s*#' "$LEVELS" | grep -v '^\s*$' | while read -r level; do
  echo "== $(date -Is) $level"
  bash "$HARNESS/run_episode.sh" "$level" "$RUN" "$MODEL" "$VARIANT" || echo "episode failed: $level"
done
python "$HARNESS/summarize.py" "$RUN" --model "claude-$MODEL-$VARIANT" --export
echo "end=$(date -Is)"
