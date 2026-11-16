#!/usr/bin/env bash
#SBATCH --job-name=luban-partial
#SBATCH --partition=researchshort
#SBATCH --account=eepengresearch
#SBATCH --qos=research-1-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/partial_%j.out
# Rerun unsolved levels that have legal moves, recording how far the search got.
cd "$SLURM_SUBMIT_DIR"
solve() {
  ./publish/linux-x64-v2/LubanSolver levels/Puzzle_$1.json --max-states 3000000 --out results_partial/Puzzle_$1.json > /dev/null 2> logs/partial_$1.err
  python3 -c "import json; d=json.load(open(\"results_partial/Puzzle_$1.json\")); d.pop(\"moves\"); d.pop(\"rule\", None); print(d, flush=True)"
}
export -f solve
echo "start $(date -Is)"
printf "%s\n" "$@" | xargs -P 3 -I{} bash -c "solve {}"
echo "end $(date -Is)"
