#!/usr/bin/env bash
#SBATCH --job-name=stack-cc
#SBATCH --partition=researchshort
#SBATCH --account=eepengresearch
#SBATCH --qos=research-1-qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
#
# Run Claude Code closed-loop episodes on Stacking levels, one after another.
#   sbatch --chdir=<run_root> run_stack_levels.sbatch <first-last> <run_root> <model> [variant]
# e.g. 0-76 for all 77 levels. No GPU: the agent runs remotely, the node only renders images.
set -uo pipefail

RANGE="$1"; RUN="$2"; MODEL="$3"; VARIANT="${4:-base}"
HARNESS="$HOME/programs/2026/CHAIN-linux/Stacking_scaling/chain_analysis/cc_harness/stacking"

source /opt/apps/software/Anaconda3/2024.06-1/etc/profile.d/conda.sh
conda activate "$HOME/envs/phyv_env"
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$RUN"
echo "node=$(hostname) start=$(date -Is) model=$MODEL variant=$VARIANT levels=$RANGE"
for level in $(seq "${RANGE%-*}" "${RANGE#*-}"); do
  echo "== $(date -Is) level $level"
  bash "$HARNESS/run_stack_episode.sh" "$level" "$RUN" "$MODEL" "$VARIANT" || echo "episode failed: $level"
done
python "$HARNESS/summarize_stack.py" "$RUN"
echo "end=$(date -Is)"
