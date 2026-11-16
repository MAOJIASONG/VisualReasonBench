#!/usr/bin/env bash
#SBATCH --job-name=exp1-agents
#SBATCH --partition=researchshort
#SBATCH --account=eepengresearch
#SBATCH --qos=research-1-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=chain_analysis/exp1_vs_agents/exp1_%j.out
# Exp 1 (dead ends, step value) on the agents' trajectories, the three games in parallel.
source /opt/apps/software/Anaconda3/2024.06-1/etc/profile.d/conda.sh
conda activate "$HOME/envs/phyv_env"
cd "$SLURM_SUBMIT_DIR"
for g in type_3 type_3_1 type_2_3; do
  python -m chain_analysis.exp1_step_value --root $g/puzzles_50 \
    --traj chain_analysis/exp1_vs_agents/traj_$g.jsonl --out chain_analysis/exp1_vs_agents/$g \
    > chain_analysis/exp1_vs_agents/log_$g.txt 2>&1 &
done
wait
echo done
