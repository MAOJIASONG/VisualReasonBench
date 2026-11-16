"""Exercise restored-rule actions and all 32 scene loads in the real Unity player."""
import argparse
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback
from live_suite import Client, copy_captures
from checks import verify_runtime_manifest
from processes import stop_owned_group


def poses(state):
    return [{k: p[k] for k in ('Id', 'X', 'Y', 'Z', 'Orientation', 'HalfRotateAxis')}
            for p in state['engine_state']['Pieces']]


def check_native_positions(state):
    pieces = state['engine_state']['Pieces']
    assert len(pieces) == len(state['native_positions'])
    for native in state['native_positions']:
        for axis in 'xyz':
            assert abs(native[axis] - float(Fraction(pieces[native['id']][axis.upper()])) / 10000) < 1e-6
            assert abs(native['visual_logical_position'][axis] - float(Fraction(pieces[native['id']][axis.upper()])) / 10000) < .001


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--levels', nargs='+', type=int, default=list(range(32)))
    args = parser.parse_args()
    package, out = args.package.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {'complete': False, 'runtime_manifest': verify_runtime_manifest(package),
              'levels': [], 'regressions': {}}
    def save():
        (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    def interrupted(*unused):
        raise KeyboardInterrupt('interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    client, proc = Client(out, timeout=report['runtime_manifest']['response_timeout_ms']/1000+10), None
    try:
        try:
            with socket.create_connection(('127.0.0.1', 9999), 1):
                raise RuntimeError('Existing Unity server; refusing to interfere')
        except ConnectionRefusedError:
            pass
        proc = subprocess.Popen(['xvfb-run', '-a', '-s', '-screen 0 1280x720x24',
                                 str(package/'LuBanSuoEnv.x86_64'), '-job-worker-count', '2',
                                 '-logFile', str(out/'unity.log')], cwd=package,
                                stdout=(out/'launcher.log').open('w'), stderr=subprocess.STDOUT,
                                start_new_session=True)
        for _ in range(180):
            if proc.poll() is not None:
                raise RuntimeError('Unity exited during startup')
            try:
                with socket.create_connection(('127.0.0.1', 9999), .5):
                    break
            except OSError:
                time.sleep(.5)
        else:
            raise TimeoutError('Unity startup')

        initial = client.success('original_start', {'level_index': 0})
        response = client.call('original_action', {'kind': 'move', 'piece_ids': [0], 'axis': 'x', 'distance': 1})
        assert response['status'] == 'fail', response
        blocked = client.success('original_state')
        assert initial == blocked, 'Blocked action changed the environment'
        report['regressions']['blocked_action_unchanged'] = True

        before = client.success('original_action', {'kind': 'move', 'piece_ids': [0], 'axis': 'z', 'distance': -300})
        half = client.success('original_action', {'kind': 'rotate', 'piece_ids': [0], 'axis': 'x', 'angle': 45})
        assert half['engine_state']['Pieces'][0]['HalfRotateAxis'] != 0
        q1, q2 = (s['native_positions'][0]['visual_rotation'] for s in [before, half])
        dot = sum(q1[k]*q2[k] for k in 'xyzw')
        angle = math.degrees(2*math.acos(min(1, abs(dot))))
        assert abs(angle-45) < .1, angle
        quarter = client.success('original_action', {'kind': 'rotate', 'piece_ids': [0], 'axis': 'x', 'angle': 45})
        assert quarter['engine_state']['Pieces'][0]['HalfRotateAxis'] == 0
        restored = client.success('original_action', {'kind': 'rotate', 'piece_ids': [0], 'axis': 'x', 'angle': -90})
        assert poses(restored) == poses(before), 'Two half turns and a reverse quarter turn failed to restore pose'
        report['regressions']['half_rotation_native_visual_degrees'] = angle
        report['regressions']['half_rotation_roundtrip'] = True

        diagonal = client.success('original_action', {'kind': 'move_vector', 'piece_ids': [0], 'vector': [1, 1, 0], 'distance': 10})
        check_native_positions(diagonal)
        restored = client.success('original_action', {'kind': 'move_vector', 'piece_ids': [0], 'vector': [-1, -1, 0], 'distance': 10})
        assert poses(restored) == poses(before), 'Vector roundtrip changed pose'
        report['regressions']['vector_roundtrip'] = True
        save()
        print('ACTION REGRESSIONS PASS', flush=True)

        for level in args.levels:
            started = time.monotonic()
            row = {'level': level, 'passed': False}
            report['levels'].append(row)
            try:
                state = client.success('original_start', {'level_index': level})
                assert state['puzzle_index'] == level
                check_native_positions(state)
                row['initial_state'] = state
                row['screenshots'] = copy_captures(client.success('capture'), package, out, level)
                row['passed'] = True
            except Exception as exc:
                row['error'] = repr(exc)
            row['seconds'] = round(time.monotonic()-started, 3)
            print('ORIGINAL LOAD', level, row['passed'], row['seconds'], row.get('error', ''), flush=True)
            save()
        report['complete'] = True
    except BaseException as exc:
        report['error'] = repr(exc)
        traceback.print_exc()
    finally:
        client.close()
        if proc:
            report['process_cleanup'] = stop_owned_group(proc)
        save()
    passed = report['complete'] and all(row['passed'] for row in report['levels'])
    print('FINAL', passed, 'levels', len(report['levels']), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
