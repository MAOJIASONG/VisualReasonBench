#!/usr/bin/env bash
#SBATCH --job-name=exp2-hard
#SBATCH --partition=researchshort
#SBATCH --account=eepengresearch
#SBATCH --qos=research-1-qos
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=chain_analysis/exp2_vs_agents/exp2_hard_%j.out
source /opt/apps/software/Anaconda3/2024.06-1/etc/profile.d/conda.sh
conda activate "$HOME/envs/phyv_env"
cd "$SLURM_SUBMIT_DIR"
python -m chain_analysis.exp2_vs_agents.run_hard_budget --max-verifies 1000 --max-actions 100000 --workers 32
