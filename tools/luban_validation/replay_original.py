"""Replay candidate actions in isolated Unity, without snapshot or victory overrides."""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback
from live_suite import Client, copy_captures
from checks import verify_runtime_manifest, exact_terminal_match, certify_scene_separation
from processes import stop_owned_group


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--candidates',type=Path,nargs='+',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();package=args.package.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    report={'complete':False,'mode':'restored_original_rules','package':str(package),'cases':[],
            'assembly_sha256':hashlib.sha256((package/'LuBanSuoEnv_Data/Managed/Assembly-CSharp.dll').read_bytes()).hexdigest()}
    manifest=verify_runtime_manifest(package)
    report['runtime_manifest']=manifest
    def save():(out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    def interrupted(*unused):raise KeyboardInterrupt('interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    client=Client(out,timeout=manifest['response_timeout_ms']/1000+10);proc=None
    try:
        try:
            with socket.create_connection(('127.0.0.1',9999),1):raise RuntimeError('Existing game server; abort')
        except ConnectionRefusedError:pass
        proc=subprocess.Popen(['xvfb-run','-a','-s','-screen 0 1280x720x24',str(package/'LuBanSuoEnv.x86_64'),'-job-worker-count','2','-logFile',str(out/'unity.log')],cwd=package,stdout=(out/'launcher.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
        for _ in range(180):
            if proc.poll() is not None:raise RuntimeError('Unity launcher exited '+str(proc.returncode))
            try:
                with socket.create_connection(('127.0.0.1',9999),.5):break
            except OSError:time.sleep(.5)
        else:raise TimeoutError('Unity startup')
        for path in args.candidates:
            candidate_bytes=path.read_bytes()
            candidate=json.loads(candidate_bytes)
            if candidate.get('status')!='candidate_solution':continue
            level=int(Path(candidate['level_file']).stem.split('_')[-1])
            row={'level':level,'candidate':str(path.resolve()),'candidate_sha256':hashlib.sha256(candidate_bytes).hexdigest(),'game_verified':False,'actions':[]};report['cases'].append(row)
            try:
                if candidate.get('engine_source_sha256','').lower()!=manifest['original_engine_source']['sha256'].lower():
                    raise AssertionError('Candidate and runtime engine sources differ; regenerate the candidate with this source version')
                rules=package/'LuBanSuoEnv_Data/StreamingAssets/OriginalRules'/f'Puzzle_{level}.json'
                actual=hashlib.sha256(rules.read_bytes()).hexdigest()
                if actual.lower()!=candidate['input_sha256'].lower():raise AssertionError('Candidate input hash differs from runtime rules')
                row['initial']=client.success('original_start',{'level_index':level})
                for action in candidate['candidate_actions']:
                    params={'kind':action['Kind'],'piece_ids':action['PieceIds'],'axis':'xyz'[action['Axis']],
                            'distance_exact':action['Distance'],'angle':action['Angle'],'rule_index':action['RuleIndex']}
                    if action.get('Pivot') is not None:params['pivot']=action['Pivot']
                    if action.get('Vector') is not None:params['vector']=action['Vector']
                    state=client.success('original_action',params)
                    row['actions'].append({'action':action,'response':state});save()
                final=client.success('original_state');row['final']=final
                expected=candidate['terminal_state']['Pieces'];got=final['engine_state']['Pieces']
                exact_terminal_match(expected,got)
                row['terminal_matches_search']=True
                for p in final['native_positions']:
                    s=got[p['id']]
                    for axis in 'xyz':
                        if abs(p[axis]-float(Fraction(s[axis.upper()]))/10000)>1e-6:raise AssertionError('Native and engine position disagree')
                        if abs(p['visual_logical_position'][axis]-float(Fraction(s[axis.upper()]))/10000)>.001:raise AssertionError('Actual scene and engine position disagree')
                row['screenshots']=copy_captures(client.success('capture'),package,out,level)
                bounds=[{'id':p['id'],**p['scene_world_bounds']} for p in final['native_positions']]
                row['scene_separation']=certify_scene_separation(bounds,[p['Id'] for p in got])
                if final.get('native_unity_is_solved') is not True:raise AssertionError('Actual Unity victory not triggered')
                row['game_verified']=True
                print('REPLAY',level,'PASS',len(row['actions']),'actions',flush=True)
            except Exception as exc:
                row['error']=repr(exc);print('REPLAY',level,'FAIL',repr(exc),flush=True)
            save()
        report['complete']=True
    except BaseException as exc:report['error']=repr(exc);traceback.print_exc()
    finally:
        client.close()
        if proc:
            report['process_cleanup']=stop_owned_group(proc)
            report['launcher_exit_code']=proc.returncode
        save()
    passed=report['complete'] and len(report['cases'])>0 and all(c['game_verified'] for c in report['cases'])
    print('FINAL',passed,'cases',len(report['cases']),flush=True)
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
