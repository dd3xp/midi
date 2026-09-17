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
import json,hashlib,datetime
import numpy as np
P=Path(__file__).resolve().parent;C=P.parent/'revision_20260915/companion'
if (P.parent/'analysis_arrays.npz').exists():C=P.parent
a=np.load(C/'analysis_arrays.npz');ts=json.loads((C/'tracks.json').read_text())
def corr(x,y,w=None):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)<3 or x.std()<1e-10 or y.std()<1e-10:return None
    if w is None:w=np.ones(len(x))
    w=np.asarray(w,float);w/=w.sum();x=x-np.sum(w*x);y=y-np.sum(w*y)
    return float(np.sum(w*x*y)/np.sqrt(np.sum(w*x*x)*np.sum(w*y*y)))
records=[];models=['mamba','s4d','bigru','instrument_ols','note_shape']
for t in ts:
    pre=t['prefix'];y=a[pre+'_amp'].astype(float);notes=a[pre+'_notes'];counts=np.zeros(len(y),int);spans=[]
    for n in notes:
        s=max(0,int(float(n[0])/t['hop_time']));e=min(len(y),int(float(n[1])/t['hop_time']))
        if e>s:counts[s:e]+=1
        if e-s>=4:spans.append((s,e))
    if counts.max()>1:continue
    ym=np.array([y[s:e].mean() for s,e in spans]);weights=np.array([e-s for s,e in spans])
    for m in models:
        p=a[pre+'_'+m].astype(float);pm=np.array([p[s:e].mean() for s,e in spans])
        records.append(dict(track=t['track'],group=t['group'],model=m,notes=len(spans),Full=corr(p,y),Note=corr(pm,ym),duration_Note=corr(pm,ym,weights)))
ids=sorted({r['track'] for r in records});assert len(ids)==133
lookup={(r['track'],r['model']):r for r in records};pairs={}
for m in models[:3]:
    for ref in models[3:]:
        diffs={k:np.array([lookup[t,m][k]-lookup[t,ref][k] for t in ids]) for k in ['Full','Note','duration_Note']}
        up=diffs['Full']>0
        pairs[m+' minus '+ref]=dict(tracks=len(ids),full_up=int(up.sum()),uniform_down=int((up&(diffs['Note']<0)).sum()),duration_down=int((up&(diffs['duration_Note']<0)).sum()),both_down=int((up&(diffs['Note']<0)&(diffs['duration_Note']<0)).sum()),mean_note_difference=float(diffs['Note'].mean()),mean_duration_note_difference=float(diffs['duration_Note'].mean()))
out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=hashlib.sha256((P/'DURATION_AUDIT_PLAN.md').read_bytes()).hexdigest(),arrays_sha256=hashlib.sha256((C/'analysis_arrays.npz').read_bytes()).hexdigest(),tracks=133,per_track=records,pairs=pairs)
target=P/'duration_weights.json'
if target.exists():
    old=json.loads(target.read_text());assert_reproduced(out, old)
    target=P/'duration_weights_recomputed.json'
target.write_text(json.dumps(out,indent=2));print(json.dumps(pairs,indent=2))
