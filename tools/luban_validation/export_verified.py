"""Export all 32 proven solution paths with their matching native-game evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    if not summary['complete'] or summary['game_verified_count'] != 32:
        raise ValueError('Export requires all 32 levels to pass real-game verification')
    if args.out.exists():
        raise ValueError('Output already exists; refusing to overwrite evidence')
    pending = []
    for row in summary['levels']:
        candidate_path = Path(row['candidate'])
        raw = candidate_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['candidate_sha256']:
            raise ValueError('Candidate changed after verification')
        receipt_path = Path(row['replay_report'])
        report = json.loads(receipt_path.read_text())
        matches = [r for r in report['cases'] if r['level'] == row['level']
                   and r.get('candidate_sha256') == row['candidate_sha256'] and r.get('game_verified') is True]
        if len(matches) != 1:
            raise ValueError('Missing or ambiguous matching game receipt')
        receipt = matches[0]
        pending.append((raw, {
            'level': row['level'], 'candidate_file': f'Puzzle_{row["level"]}.json',
            'candidate_sha256': row['candidate_sha256'], 'action_count': row['actions'],
            'game_verified': True, 'terminal_matches_search': receipt['terminal_matches_search'],
            'native_unity_is_solved': receipt['final']['native_unity_is_solved'],
            'actual_scene_separation': receipt['scene_separation'],
            'actual_native_positions': receipt['final']['native_positions'],
            'engine_source_sha256': row['engine_source_sha256'],
            'runtime_assembly_sha256': report['runtime_manifest']['output_sha256'],
            'runtime_extension_sha256': report['runtime_manifest']['extension_sha256'],
            'full_replay_report_sha256': hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        }))
    if sorted(r['level'] for _, r in pending) != list(range(32)):
        raise ValueError('Need exactly the 32 distinct levels')
    args.out.mkdir(parents=True)
    for raw, row in pending:
        (args.out/row['candidate_file']).write_bytes(raw)
    index = {'schema_version': 1, 'all_32_game_verified': True,
             'note': 'Candidate files retain their original model-only labels and hashes. This index records separate actual Unity verification. No shortest-path claim.',
             'levels': [row for _, row in pending]}
    (args.out/'verification_index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2)+'\n')
    print('Exported 32 verified solutions to', args.out)


if __name__ == '__main__':
    main()
