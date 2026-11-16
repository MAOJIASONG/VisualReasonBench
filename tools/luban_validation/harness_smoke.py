"""Replay a candidate through the project's normal environment and observations."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
from phyvpuzzle.core.base import Action
from phyvpuzzle.environment.luban_env import LubanConfig, LubanEnvironment
from checks import verify_runtime_manifest, exact_terminal_match


def tool_action(action):
    kind, ids = action['Kind'], action['PieceIds']
    if kind == 'special':
        return Action('apply_special_rule', {'rule_index': action['RuleIndex']})
    if kind == 'move_vector':
        return Action('move_vector', {'piece_ids': ids, 'vector': action['Vector'], 'distance': action['Distance']})
    params = {'axis': 'xyz'[action['Axis']]}
    if kind == 'move':
        params['distance'] = action['Distance']
    elif kind == 'rotate':
        params['angle'] = action['Angle']
        if action.get('Pivot') is not None:
            params['pivot'] = action['Pivot']
    else:
        raise ValueError('Unsupported candidate action')
    params['piece_id' if len(ids) == 1 else 'piece_ids'] = ids[0] if len(ids) == 1 else ids
    return Action(kind+'_piece' if len(ids) == 1 else kind+'_group', params)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    package = args.package.resolve()
    manifest = verify_runtime_manifest(package)
    candidate_bytes = args.candidate.read_bytes()
    candidate = json.loads(candidate_bytes)
    assert candidate['status'] == 'candidate_solution'
    assert candidate['engine_source_sha256'].lower() == manifest['original_engine_source']['sha256'].lower()
    level = int(Path(candidate['level_file']).stem.split('_')[-1])
    rules = package/'LuBanSuoEnv_Data/StreamingAssets/OriginalRules'/f'Puzzle_{level}.json'
    assert hashlib.sha256(rules.read_bytes()).hexdigest().lower() == candidate['input_sha256'].lower()
    try:
        with socket.create_connection(('127.0.0.1', 9999), 1):
            raise RuntimeError('Existing server; this smoke must test automatic startup')
    except ConnectionRefusedError:
        pass
    environment = LubanEnvironment(LubanConfig(type='luban', action_mode='restored_original_rules',
        exe_path=str(package/'LuBanSuoEnv.x86_64'), level_index=level))
    report = {'passed': False, 'level': level, 'candidate_sha256': hashlib.sha256(candidate_bytes).hexdigest(),
              'runtime_manifest': manifest, 'steps': []}
    def save():
        (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    def record(observation):
        index = len(report['steps'])
        observation.image.save(out/f'observation_{index:02d}.jpg')
        assert observation.image.width > 0 and observation.image.height > 0
        row = {'state': observation.state.to_dict(), 'description': observation.description,
               'image_size': list(observation.image.size)}
        report['steps'].append(row)
        save()
    try:
        started = time.monotonic()
        observation = environment.reset()
        report['reset_seconds'] = time.monotonic()-started
        assert environment._owns_process, 'Automatic launch was not exercised'
        report['automatically_launched'] = True
        report['tools'] = [s['function']['name'] for s in environment.get_tool_schemas()]
        record(observation)
        for spec in candidate['candidate_actions']:
            observation = environment.step(tool_action(spec))
            record(observation)
            assert observation.state.metadata['tool_result']['status'] == 'success'
        assert observation.state.metadata['is_solved'] is True
        assert observation.state.metadata['scene_separated'] is True
        exact_terminal_match(candidate['terminal_state']['Pieces'], observation.state.metadata['engine_state']['Pieces'])
        report['passed'] = True
    except BaseException as exc:
        report['error'] = repr(exc)
        traceback.print_exc()
    finally:
        environment.close()
        # xvfb-run's shell can exit just before Unity finishes graceful shutdown.
        report['owned_server_stopped'] = False
        for _ in range(50):
            try:
                with socket.create_connection(('127.0.0.1', 9999), .1):
                    pass
            except ConnectionRefusedError:
                report['owned_server_stopped'] = True
                break
            time.sleep(.1)
        if not report['owned_server_stopped']:
            report['passed'] = False
        save()
    print('HARNESS', report['passed'], 'level', level, 'steps', len(report['steps'])-1, flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
