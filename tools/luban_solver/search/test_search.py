#!/usr/bin/env python3
"""Integration regressions: resource caps, exact resume, model replay and stale receipts."""
import argparse, json, pathlib, subprocess, tempfile

p=argparse.ArgumentParser();p.add_argument('--dotnet',default='dotnet');p.add_argument('--levels',required=True);p.add_argument('--dll');a=p.parse_args()
dll=pathlib.Path(a.dll) if a.dll else pathlib.Path(__file__).parent/'bin/Release/net8.0/LubanSearch.dll'
level=pathlib.Path(a.levels).resolve()/'Puzzle_0.json'
def run(args,expected):
 r=subprocess.run([a.dotnet,str(dll),*map(str,args)],capture_output=True,text=True,timeout=90)
 assert r.returncode==expected,(r.returncode,r.stdout,r.stderr)
 return r
with tempfile.TemporaryDirectory(prefix='luban-search-tests-') as t:
 t=pathlib.Path(t);limited=t/'limited.json';full=t/'full.json'
 common=['--seconds','30','--max-depth','20','--expansion-batch','2']
 run([level,'--out',limited,'--max-states','2',*common],3)
 d=json.loads(limited.read_text());assert d['status']=='state_limit';assert d['states_seen']==2
 run([level,'--out',limited,'--resume',str(limited)+'.checkpoint.json','--max-states','1000',*common],0)
 resumed=json.loads(limited.read_text());assert resumed['status']=='candidate_solution'
 run([level,'--out',full,'--max-states','1000',*common],0)
 plain=json.loads(full.read_text());assert resumed['candidate_actions']==plain['candidate_actions'];assert resumed['terminal_state']==plain['terminal_state']
 assert not plain['game_verified'];assert plain['excludes_whole_assembly_move']
 run(['replay',level,full],0)
 replay=json.loads(pathlib.Path(str(full)+'.model-replay.json').read_text());assert replay['model_replay_passed'];assert not replay['game_verified']
 stale=t/'stale-checkpoint.json';cp=json.loads(pathlib.Path(str(full)+'.checkpoint.json').read_text());cp['EngineHash']='bad';stale.write_text(json.dumps(cp))
 run([level,'--out',full,'--resume',stale,'--max-states','1000',*common],2)
 changed=t/'changed.json';d=json.loads(level.read_text());d['polyhedronBaseIndex']=1-d['polyhedronBaseIndex'];changed.write_text(json.dumps(d))
 run(['replay',changed,full],2)
 stale_receipt=t/'stale-source.json';bad=dict(plain);bad['engine_source_sha256']='bad';stale_receipt.write_text(json.dumps(bad));run(['replay',level,stale_receipt],2)
 actions=t/'actions.json';actions.write_text(json.dumps(plain['candidate_actions']));certified=t/'certified.json';run(['certify',level,actions,certified],0)
 assert json.loads(certified.read_text())['candidate_origin']=='provided_path_revalidated'
 seeded=t/'seeded.json';run([level,'--initial-actions',actions,'--out',seeded,*common],0)
 assert json.loads(seeded.read_text())['candidate_actions']==plain['candidate_actions']
 run([level,'--seconds','NaN','--out',t/'nan.json'],2)
 deadline=t/'deadline.json';run([pathlib.Path(a.levels).resolve()/'Puzzle_3.json','--seconds','0.002','--groups','all','--out',deadline],3);assert json.loads(deadline.read_text())['status']=='time_limit'
 too_long=t/'too-long.json';too_long.write_text(json.dumps(plain['candidate_actions']*2));run([level,'--initial-actions',too_long,'--max-depth','1','--out',t/'too-long-out.json'],2)
 parked=t/'parked.json';run([level,'--park-finished','--park-components','--out',parked,*common],0);run(['replay',level,parked],0)
 # At least one repaired puzzle must require and preserve the linked special action.
 repaired=t/'level4.json';run([pathlib.Path(a.levels).resolve()/'Puzzle_4.json','--out',repaired,'--max-states','20000','--seconds','30'],0)
 d=json.loads(repaired.read_text());assert any(x['Kind']=='special' for x in d['candidate_actions']);assert d['game_verified'] is False
 run(['replay',pathlib.Path(a.levels).resolve()/'Puzzle_4.json',repaired],0)
 # Preserve a partially enumerated parent when a promising child is explored.
 partial=t/'partial.json';level4=pathlib.Path(a.levels).resolve()/'Puzzle_4.json'
 batch=['--seconds','30','--expansion-batch','16']
 run([level4,'--out',partial,'--max-states','80',*batch],3)
 cp=json.loads(pathlib.Path(str(partial)+'.checkpoint.json').read_text());assert any(n['ResumeAction']>0 for n in cp['Nodes'])
 run([level4,'--out',partial,'--resume',str(partial)+'.checkpoint.json','--max-states','20000',*batch],0)
 fresh=t/'batch-fresh.json';run([level4,'--out',fresh,'--max-states','20000',*batch],0)
 assert json.loads(partial.read_text())['candidate_actions']==json.loads(fresh.read_text())['candidate_actions']
print('PASS: state cap, incremental exact resume, fresh model replay, input/model/source mismatch rejection, supplied-path provenance, validated prefix, original special action path.')
