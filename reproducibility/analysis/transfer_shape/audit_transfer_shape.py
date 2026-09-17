def assert_reproduced(actual, expected, path='root'):
    """Exact structure/count/hash checks; float-only 1e-12 platform tolerance."""
    import math
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            if key != 'utc': assert_reproduced(actual[key], expected[key], path+'/'+key)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected)): assert_reproduced(a, e, path+'/'+str(i))
    elif isinstance(expected, float):
        assert math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (path, actual, expected)
    else:
        assert actual == expected, (path, actual, expected)

from pathlib import Path
import sys,json,hashlib,datetime
import numpy as np
P=Path(__file__).resolve().parent
C=P/'musicnet_companion'
if not C.exists():C=P.parent/'musicnet'
sys.path.insert(0,str(C))
from metrics_core import corr,metrics
a=np.load(C/'inputs.npz');pred=np.load(C/'predictions_all.npz')
tracks=json.loads((C/'tracks.json').read_text());expected=json.loads((C/'results.json').read_text())
models=sorted(expected['summary']);records=[];coverage=[]
for t in tracks:
    pre=t['track'].split('/')[-1];y=a[pre+'_amp'].astype(float);notes=a[pre+'_notes'];h=t['hop_time']
    counts=np.zeros(len(y),int);spans=[]
    for n in notes:
        s=max(0,int(float(n[0])/h));e=min(len(y),int(float(n[1])/h))
        if e>s:counts[s:e]+=1
        if e-s>=4:spans.append((s,e))
    clean=[(s,e) for s,e in spans if np.all(counts[s:e]<=1)]
    coverage.append(dict(track=t['track'],group=t['group'],eligible=len(spans),nonoverlap=len(clean),fully_nonoverlap=bool(counts.max()<=1)))
    for m in models:
        p=pred[pre+'_'+m].astype(float)
        within=[corr(p[s:e],y[s:e]) for s,e in clean];within=[v for v in within if v is not None]
        mu=np.array([y[s:e].mean() for s,e in clean]);mp=np.array([p[s:e].mean() for s,e in clean])
        old=next(r for r in expected['per_track'] if r['track']==t['track'] and r['model']==m)
        records.append(dict(track=t['track'],group=t['group'],model=m,eligible=len(clean),defined=len(within),
            full_original=corr(p,y),original_defined=old['defined_shape_notes'],
            Within=float(np.mean(within)) if within else None,Note=corr(mp,mu),
            Error=float(np.sqrt(np.mean((np.log(mp+1e-7)-np.log(mu+1e-7))**2))) if len(mu) else None))
summary={}
for m in models:
    rs=[r for r in records if r['model']==m]
    summary[m]=dict(original_defined_notes=sum(r['original_defined'] for r in rs),nonoverlap_defined_notes=sum(r['defined'] for r in rs))
    for k in ['Within','Note','Error']:
        vals=[r[k] for r in rs if r[k] is not None]
        summary[m][k]=dict(n=len(vals),mean=float(np.mean(vals)) if vals else None)
ids={c['track'] for c in coverage if c['fully_nonoverlap']}
out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=hashlib.sha256((P/'TRANSFER_SHAPE_AUDIT_PLAN.md').read_bytes()).hexdigest(),
    inputs_sha256=hashlib.sha256((C/'inputs.npz').read_bytes()).hexdigest(),predictions_sha256=hashlib.sha256((C/'predictions_all.npz').read_bytes()).hexdigest(),
    recordings=len(tracks),eligible_notes=sum(c['eligible'] for c in coverage),nonoverlap_notes=sum(c['nonoverlap'] for c in coverage),
    fully_nonoverlap_recordings=sorted(ids),complete_horizon_full={m:float(np.mean([r['full_original'] for r in records if r['model']==m and r['track'] in ids])) if ids else None for m in models},
    coverage=coverage,summary=summary,per_track=records)
target=P/'transfer_shape.json'
if target.exists():
    old=json.loads(target.read_text());assert_reproduced(out, old)
    target=P/'transfer_shape_recomputed.json'
target.write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps({k:v for k,v in out.items() if k not in ['per_track','coverage']},indent=2))
