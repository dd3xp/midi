from pathlib import Path
import datetime,hashlib,json
import numpy as np
P=Path(__file__).resolve().parent;C=P.parent if (P.parent/'analysis_arrays.npz').exists() else P.parent/'revision_20260915/companion'
meta=json.loads((C/'tracks.json').read_text());old=json.loads((C/'expected_results.json').read_text());arrays=np.load(C/'analysis_arrays.npz');models=list(old['per_track']);thresholds=[4,10,20,50]
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'thresholds':thresholds,'per_track':{},'summary':{},'paired':{},'sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [P/'DURATION_PLAN.md',P/'duration_check.py',C/'analysis_arrays.npz',C/'tracks.json',C/'expected_results.json']}}
for cut in thresholds:out['per_track'][str(cut)]={m:[] for m in models}
maxerr=0.;checks=0
for row in meta:
 pre=row['prefix'];y=arrays[pre+'_amp'].astype(float);h=row['hop_time'];spans=[(max(0,int(float(n[0])/h)),min(len(y),int(float(n[1])/h))) for n in arrays[pre+'_notes']];spans=[(s,e) for s,e in spans if e-s>=4];lens=np.array([e-s for s,e in spans])
 for m in models:
  p=arrays[pre+'_'+m].astype(float);cs=[]
  for s,e in spans:
   x=p[s:e];z=y[s:e]
   if x.std()<1e-10 or z.std()<1e-10:cs.append(np.nan);continue
   dx=x-x.mean();dz=z-z.mean();cs.append(np.clip(np.dot(dx,dz)/np.sqrt(np.dot(dx,dx)*np.dot(dz,dz)),-1,1))
  cs=np.array(cs)
  for cut in thresholds:
   use=(lens>=cut)&np.isfinite(cs);v=float(cs[use].mean()) if use.any() else None
   out['per_track'][str(cut)][m].append({'track':row['track'],'within':v,'eligible_notes':int((lens>=cut).sum()),'defined_notes':int(use.sum())})
   if cut==4:
    orig=next(t for t in old['per_track'][m] if t['track']==row['track'])['views']['all']['within_note']
    assert (orig is None)==(v is None)
    if v is not None:maxerr=max(maxerr,abs(v-orig));assert abs(v-orig)<1e-10
    checks+=1
def paired(v):
 ix=np.flatnonzero(np.isfinite(v));vs=v[ix];gs=sorted({meta[i]['group'] for i in ix});labels=[gs.index(meta[i]['group']) for i in ix]
 draws=np.random.RandomState(20260911).randint(len(gs),size=(5000,len(gs)));w=np.stack([(draws==j).sum(1) for j in range(len(gs))],1)[:,labels];boots=w@vs/w.sum(1)
 return {'mean':float(vs.mean()),'ci95':np.quantile(boots,[.025,.975]).tolist(),'tracks':len(ix),'groups':len(gs)}
for cut in map(str,thresholds):
 out['summary'][cut]={};out['paired'][cut]={}
 for m in models:
  rows=out['per_track'][cut][m];vs=[x['within'] for x in rows if x['within'] is not None]
  out['summary'][cut][m]={'mean':float(np.mean(vs)) if vs else None,'defined_tracks':len(vs),'defined_notes':sum(x['defined_notes'] for x in rows),'eligible_notes':sum(x['eligible_notes'] for x in rows)}
 for m in ['mamba','s4d','bigru']:
  for ref in ['instrument_ols','note_shape']:
   vals=[np.nan if a['within'] is None or b['within'] is None else a['within']-b['within'] for a,b in zip(out['per_track'][cut][m],out['per_track'][cut][ref])]
   out['paired'][cut][m+' minus '+ref]=paired(np.array(vals))
out['verification']={'original_within_max_error':maxerr,'original_model_track_checks':checks}
(P/'duration_results.json').write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k in ['summary','paired','verification']},indent=2))
