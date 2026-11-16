#!/usr/bin/env bash
#SBATCH --job-name=luban-solve
#SBATCH --partition=researchshort
#SBATCH --account=eepengresearch
#SBATCH --qos=research-1-qos
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/solve_all_%j.out
# Solve every level in one job, 8 levels at a time:  sbatch run_all.sbatch [max_states]
cd "$SLURM_SUBMIT_DIR"
export MAXS="${1:-3000000}"
solve() {
  ./publish/linux-x64/LubanSolver levels/Puzzle_$1.json --max-states "$MAXS" --out results/Puzzle_$1.json > /dev/null 2> logs/solve_$1.err
  python3 -c "import json; d=json.load(open('results/Puzzle_$1.json')); d.pop('moves'); d.pop('rule', None); print(d, flush=True)"
}
export -f solve
echo "start $(date -Is) max_states=$MAXS"
seq 0 31 | xargs -P 8 -I{} bash -c 'solve {}'
echo "end $(date -Is)"
