#!/usr/bin/env python3
"""Run bounded per-level searches, preserving every receipt and process exit status."""
import argparse, concurrent.futures, json, os, pathlib, subprocess, time

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dotnet',default='dotnet');p.add_argument('--levels',default='0-31')
    p.add_argument('--input-dir',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--seconds',type=float,default=60);p.add_argument('--max-states',type=int,default=100000)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--max-depth',type=int,default=80)
    p.add_argument('--strategy',default='best-first',choices=['bfs','best-first'])
    p.add_argument('--groups',default='closure',choices=['singles','closure','all'])
    p.add_argument('--distances',default='events',choices=['events','grid'])
    p.add_argument('--expansion-batch',type=int,default=0);p.add_argument('--park-finished',action='store_true');p.add_argument('--park-components',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--dll');p.add_argument('--directions',default='axes',choices=['axes','grid26'])
    a=p.parse_args()
    levels=[]
    for part in a.levels.split(','):
        if '-' in part:
            lo,hi=map(int,part.split('-'));levels.extend(range(lo,hi+1))
        else:levels.append(int(part))
    out=pathlib.Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    dll=pathlib.Path(a.dll).resolve() if a.dll else pathlib.Path(__file__).parent/'bin/Release/net8.0/LubanSearch.dll'
    def run(level):
        dest=out/f'Puzzle_{level}.json';cmd=[a.dotnet,str(dll),str(pathlib.Path(a.input_dir).resolve()/f'Puzzle_{level}.json'),'--out',str(dest),'--seconds',str(a.seconds),'--max-states',str(a.max_states),'--max-depth',str(a.max_depth),'--strategy',a.strategy,'--groups',a.groups,'--distances',a.distances,'--directions',a.directions,'--expansion-batch',str(a.expansion_batch)]
        if a.park_finished:cmd+=['--park-finished']
        if a.park_components:cmd+=['--park-components']
        checkpoint=pathlib.Path(str(dest)+'.checkpoint.json')
        if a.resume and checkpoint.exists():cmd+=['--resume',str(checkpoint)]
        receipt={'level':level,'command':cmd,'started_unix':time.time()};started=time.monotonic()
        try:
            with open(out/f'Puzzle_{level}.stdout','a' if a.resume else 'w') as stdout,open(out/f'Puzzle_{level}.stderr','a' if a.resume else 'w') as stderr:
                r=subprocess.run(cmd,stdout=stdout,stderr=stderr,timeout=a.seconds+180,env={**os.environ,'DOTNET_CLI_TELEMETRY_OPTOUT':'1'})
            receipt['exit_code']=r.returncode
        except subprocess.TimeoutExpired:receipt['exit_code']=None;receipt['external_timeout']=True
        receipt['wall_seconds']=round(time.monotonic()-started,3)
        if dest.exists():
            d=json.loads(dest.read_text());receipt.update({k:d.get(k) for k in ['status','states_seen','actions_tried','game_verified']});receipt['candidate_length']=len(d.get('candidate_actions') or [])
        (out/f'Puzzle_{level}.process.json').write_text(json.dumps(receipt,indent=2))
        print(json.dumps({k:v for k,v in receipt.items() if k not in ['command','started_unix']}),flush=True)
        return receipt
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(run,levels))
    (out/'summary.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':main()
