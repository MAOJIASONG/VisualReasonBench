#!/usr/bin/env bash
#SBATCH --job-name=stk-oracle
#SBATCH --partition=researchshort
#SBATCH --account=eepengresearch
#SBATCH --qos=research-1-qos
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out
# Stacking oracle: level trap statistics (Exp 1) and search baselines (Exp 2), all 77 levels.
#   sbatch --chdir=<Stacking_scaling>/chain_analysis/stacking_oracle run.sbatch
set -uo pipefail
source /opt/apps/software/Anaconda3/2024.06-1/etc/profile.d/conda.sh
conda activate "$HOME/envs/phyv_env"
cd "$HOME/programs/2026/CHAIN-linux/Stacking_scaling"
OUT=chain_analysis/stacking_oracle/results; mkdir -p $OUT
echo "start $(date -Is) $(hostname)"
python -m chain_analysis.stacking_oracle.exp2_stacking run --out $OUT/runs.jsonl --seeds 5 --cap 1000 --workers 31 &
python -m chain_analysis.stacking_oracle.exp1_stacking --out $OUT/exp1 --levels-only
wait
echo "end $(date -Is)"
