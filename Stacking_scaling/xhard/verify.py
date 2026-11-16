"""Verify the selected assets and replay every solution with the real game rules.

Run from the repository root: python Stacking_scaling/xhard/verify.py
This maintainer check reads answers; never expose it to an evaluated model.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
STACKING = HERE.parent
sys.path.insert(0, str(STACKING))
from chain_analysis.session import replay, terminal


def verify():
    manifest = json.loads((HERE / 'selection.json').read_text())
    levels = manifest['levels']
    expected = {'fluidnet': 5, 'pipefit': 5, 'domino': 5}
    if Counter(row['game'] for row in levels) != expected:
        raise ValueError('Selection must contain exactly five levels per game')
    if len({row['path'] for row in levels}) != 15:
        raise ValueError('Duplicate level in selection')
    results = []
    for row in levels:
        directory = STACKING / row['path']
        required = {'level.json', 'solution.txt', '00_initial_front.png', '00_initial_back.png'}
        if set(row['sha256']) != required:
            raise ValueError(f"Missing asset receipt: {row['id']}")
        for name, digest in row['sha256'].items():
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Asset differs from selection: {directory / name}')
        actions = [json.loads(line) for line in (directory / 'solution.txt').read_text().splitlines()
                   if line.startswith('[')]
        if not actions or len(actions) != row['solution_actions'] or len(actions) > 40:
            raise ValueError(f"Invalid solution length: {row['id']}")
        codes = []
        for index in range(1, len(actions) + 1):
            state, last = replay(row['game'], str(directory), actions[:index])
            code = last.code.value.upper() if last is not None else ''
            if code not in {'OK', 'WIN'}:
                raise ValueError(f"{row['id']} action {index} rejected: {code}")
            if terminal(state, last) and index != len(actions):
                raise ValueError(f"{row['id']} has actions after the game ended")
            codes.append(code)
        if terminal(state, last) != 'win':
            raise ValueError(f"Solution did not win: {row['id']}")
        results.append(dict(id=row['id'], game=row['game'], path=row['path'],
                            actions=len(actions), status='win', action_codes=codes))
    return dict(levels_checked=len(results), wins=len(results),
                selection_sha256=hashlib.sha256((HERE / 'selection.json').read_bytes()).hexdigest(),
                results=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, help='Optionally save the validation report')
    args = parser.parse_args()
    report = verify()
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(f"Verified assets and winning solutions: {report['wins']}/{report['levels_checked']}")
