"""Join solution files to matching real-game replay receipts; never infer success."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solutions', type=Path, required=True)
    parser.add_argument('--replays', type=Path, nargs='+', required=True)
    parser.add_argument('--loads', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    receipts = []
    for path in args.replays:
        report = json.loads(path.read_text())
        for row in report['cases']:
            if (report['complete'] and row.get('game_verified') is True
                    and row.get('terminal_matches_search') is True
                    and row.get('scene_separation', {}).get('all_pairs_separated') is True
                    and row.get('final', {}).get('native_unity_is_solved') is True):
                receipts.append((path, report, row))
    loads = json.loads(args.loads.read_text())
    loaded = {row['level'] for row in loads['levels'] if row.get('passed') is True}
    rows = []
    for level in range(32):
        path = args.solutions/f'Puzzle_{level}.json'
        row = {'level': level, 'load_and_capture_verified': level in loaded,
               'candidate_found': False, 'game_verified': False}
        if path.exists():
            candidate_bytes = path.read_bytes()
            candidate = json.loads(candidate_bytes)
            if candidate.get('status') == 'candidate_solution':
                digest = hashlib.sha256(candidate_bytes).hexdigest()
                row.update(candidate_found=True, candidate=str(path.resolve()), candidate_sha256=digest,
                           actions=len(candidate['candidate_actions']),
                           engine_source_sha256=candidate['engine_source_sha256'])
                for receipt_path, report, receipt in receipts:
                    if (receipt['level'] == level and receipt.get('candidate_sha256') == digest
                            and report['runtime_manifest']['original_engine_source']['sha256'].lower()
                            == candidate['engine_source_sha256'].lower()):
                        row.update(game_verified=True, replay_report=str(receipt_path.resolve()),
                                   separated_pairs=len(receipt['scene_separation']['pairs']),
                                   native_unity_is_solved=True)
                        break
        rows.append(row)
    summary = {'complete': loads['complete'] and len(loaded) == 32 and all(r['game_verified'] for r in rows),
               'loaded_and_rendered': len(loaded), 'candidate_count': sum(r['candidate_found'] for r in rows),
               'game_verified_count': sum(r['game_verified'] for r in rows),
               'load_report': str(args.loads.resolve()), 'levels': rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ['levels', 'load_report']}))


if __name__ == '__main__':
    main()
