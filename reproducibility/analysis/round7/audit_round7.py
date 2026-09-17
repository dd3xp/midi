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
import numpy as np,json,hashlib,datetime
P=Path(__file__).resolve().parent;C=P.parent/'revision_20260915/companion'
if (P.parent/'analysis_arrays.npz').exists():C=P.parent
ts=json.loads((C/'tracks.json').read_text());a=np.load(C/'analysis_arrays.npz')
models=['gate','global_ols','instrument_ols','note_ols','note_shape','mamba','s4d','bigru']
def corr(x,y):
 return None if len(x)<3 or np.std(x)<1e-10 or np.std(y)<1e-10 else float(np.corrcoef(x,y)[0,1])
out=[]
for t in ts:
 pre=t['prefix'];y=a[pre+'_amp'].astype(float);hop=t['hop_time'];n=len(y)
 spans=[(max(0,int(float(v[0])/hop)),min(n,int(float(v[1])/hop))) for v in a[pre+'_notes']];spans=[(s,e) for s,e in spans if e-s>=4]
 keep=[(s,e) for s,e in spans if s>=2 and e<=n-2]
 ym=np.array([y[s:e].mean() for s,e in spans]);yk=np.array([y[s:e].mean() for s,e in keep]);r=dict(track=t['track'],group=t['group'],notes=len(spans),interior_notes=len(keep),models={})
 for m in models:
  v=a[pre+'_'+m].astype(float);vm=np.array([v[s:e].mean() for s,e in spans]);vk=np.array([v[s:e].mean() for s,e in keep]);d=np.log(vm+1e-7)-np.log(ym+1e-7)
  r['models'][m]=dict(Note=corr(vm,ym),Error=float(np.sqrt(np.mean(d*d))),b2=float(np.mean(d)**2),c2=float(np.var(d)),Full=corr(v,y),interior_Full=corr(v[2:-2],y[2:-2]),interior_Note=corr(vk,yk))
 out.append(r)
def summarize(values):
 values=[v for v in values if v is not None]
 return dict(n=len(values),mean=float(np.mean(values)) if values else None)
summary={m:{k:summarize([r['models'][m][k] for r in out]) for k in ['Note','Error','b2','c2']} for m in models}
pairs={}
for m,ref in [('mamba','instrument_ols'),('note_shape','note_ols')]:
 rr={}
 for prefix in ['', 'interior_']:
  f=np.array([r['models'][m][prefix+'Full']-r['models'][ref][prefix+'Full'] for r in out]);no=np.array([r['models'][m][prefix+'Note']-r['models'][ref][prefix+'Note'] for r in out]);rr[prefix or 'original']=dict(Full=float(f.mean()),Note=float(no.mean()),Full_up=int((f>0).sum()),Full_up_Note_down=int(((f>0)&(no<0)).sum()))
 pairs[m+' minus '+ref]=rr
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=hashlib.sha256((P/'ROUND7_FIXED_OUTPUT_PLAN.md').read_bytes()).hexdigest(),arrays_sha256=hashlib.sha256((C/'analysis_arrays.npz').read_bytes()).hexdigest(),tracks=186,notes=sum(r['notes'] for r in out),interior_notes=sum(r['interior_notes'] for r in out),summary=summary,pairs=pairs,per_track=out)
f=P/'round7_fixed_outputs.json'
if f.exists():
 old=json.loads(f.read_text());assert_reproduced(result, old);f=P/'round7_fixed_outputs_recomputed.json'
f.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='per_track'},indent=2))
