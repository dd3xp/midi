"""Replay the frozen control analysis in a fresh directory using supplied caches."""
from pathlib import Path
import argparse,datetime,json,shutil,subprocess,sys
import numpy as np
P=Path(__file__).resolve().parent
default=P.parent if (P.parent/'analysis_arrays.npz').exists() else P.parent/'analysis_companion'
parser=argparse.ArgumentParser()
parser.add_argument('--inputs',type=Path,default=default)
parser.add_argument('--output',type=Path,default=Path('runs')/('neural_level_control_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
cache=args.output/'revision_20260915/companion';cache.mkdir(parents=True)
for name in ['analysis_arrays.npz','tracks.json','folds.json','manifest.json']:
    shutil.copy2(args.inputs/name,cache/name)
work=args.output/'strong_level_control_20260918';work.mkdir()
for name in ['PLAN.md','prepare.py','analyze.py','verify.py']:
    shutil.copy2(P/name,work/name)
commands=[]
for name in ['prepare.py','analyze.py','verify.py']:
    command=[sys.executable,str((work/name).resolve())];commands.append(command)
    run=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (args.output/(name+'.log')).write_bytes(run.stdout)
    if run.returncode:
        print(run.stdout.decode('utf-8',errors='replace'));raise SystemExit(run.returncode)
actual=json.loads((work/'results.json').read_text());expected=json.loads((P/'results.json').read_text())
def compare(x,y,path='root'):
    if isinstance(y,dict):
        assert x.keys()==y.keys(),path
        for k in y:compare(x[k],y[k],path+'/'+k)
    elif isinstance(y,list):
        assert len(x)==len(y),path
        for i,(xx,yy) in enumerate(zip(x,y)):compare(xx,yy,path+'/'+str(i))
    elif isinstance(y,float):assert np.isclose(x,y,rtol=0,atol=1e-12),path
    else:assert x==y,path
for key in ['summary','per_track','selected_tracks','groups','input_sha256','bootstrap']:
    compare(actual[key],expected[key],key)
record=dict(status='passed',commands=commands,output=str(args.output.resolve()),per_track_and_summary_match=True,tolerance=1e-12)
(args.output/'reproduction.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
