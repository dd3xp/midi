"""Independently check every round-seven result from cached arrays on CPU."""
from pathlib import Path
import json,hashlib
import numpy as np
P=Path(__file__).resolve().parent;C=P.parent
expected=json.loads((P/'round7_fixed_outputs.json').read_text())
assert hashlib.sha256((C/'analysis_arrays.npz').read_bytes()).hexdigest()==expected['arrays_sha256']
assert hashlib.sha256((P/'ROUND7_FIXED_OUTPUT_PLAN.md').read_bytes()).hexdigest()==expected['plan_sha256']
tracks=json.loads((C/'tracks.json').read_text());arrays=np.load(C/'analysis_arrays.npz')
models=list(expected['summary']);checked=0;max_error=0.;rows=[]
def check(actual,want,label):
 global checked,max_error
 checked+=1
 if want is None:assert actual is None,label;return
 assert actual is not None,label
 err=abs(actual-want);max_error=max(max_error,err)
 assert np.isclose(actual,want,rtol=1e-8,atol=1e-8),(label,actual,want)
def pearson(x,y):
 if len(x)<3:return None
 x=x-x.mean();y=y-y.mean();xx=float(np.dot(x,x));yy=float(np.dot(y,y))
 if np.sqrt(xx/len(x))<1e-10 or np.sqrt(yy/len(y))<1e-10:return None
 return float(np.dot(x,y)/np.sqrt(xx*yy))
def interval_means(x,spans):
 acc=np.r_[0.,np.cumsum(x,dtype=np.float64)]
 return (acc[spans[:,1]]-acc[spans[:,0]])/(spans[:,1]-spans[:,0])
assert len(tracks)==len(expected['per_track'])==expected['tracks']==186
for track,want in zip(tracks,expected['per_track']):
 assert track['track']==want['track'] and track['group']==want['group']
 prefix=track['prefix'];target=arrays[prefix+'_amp'].astype(np.float64);n=len(target)
 edges=np.trunc(arrays[prefix+'_notes'][:,:2].astype(np.float64)/track['hop_time']).astype(np.int64)
 edges[:,0]=np.maximum(edges[:,0],0);edges[:,1]=np.minimum(edges[:,1],n)
 spans=edges[(edges[:,1]-edges[:,0])>=4]
 inside=spans[(spans[:,0]>=2)&(spans[:,1]<=n-2)]
 assert len(spans)==want['notes'] and len(inside)==want['interior_notes']
 truth=interval_means(target,spans);guard_truth=interval_means(target,inside);values={}
 for model in models:
  pred=arrays[prefix+'_'+model].astype(np.float64)
  assert len(pred)==n and np.isfinite(pred).all() and np.isfinite(target).all()
  means=interval_means(pred,spans);delta=np.log(means+1e-7)-np.log(truth+1e-7)
  offset=float(np.sum(delta)/len(delta));centered=delta-offset
  result=dict(Note=pearson(means,truth),Error=float(np.linalg.norm(delta)/np.sqrt(len(delta))),b2=offset*offset,c2=float(np.dot(centered,centered)/len(delta)),Full=pearson(pred,target),interior_Full=pearson(pred[2:n-2],target[2:n-2]),interior_Note=pearson(interval_means(pred,inside),guard_truth))
  assert np.isclose(result['Error']**2,result['b2']+result['c2'],rtol=1e-12,atol=1e-12)
  for key,value in result.items():check(value,want['models'][model][key],(track['track'],model,key))
  values[model]=result
 rows.append(values)
assert sum(x['notes'] for x in expected['per_track'])==expected['notes']
assert sum(x['interior_notes'] for x in expected['per_track'])==expected['interior_notes']
for model,metrics in expected['summary'].items():
 for metric,want in metrics.items():
  values=[r[model][metric] for r in rows if r[model][metric] is not None]
  assert len(values)==want['n']
  check(float(np.mean(values)) if values else None,want['mean'],(model,metric,'summary'))
for pair,conditions in expected['pairs'].items():
 model,reference=pair.split(' minus ')
 for condition,want in conditions.items():
  prefix='' if condition=='original' else condition
  full=np.array([r[model][prefix+'Full']-r[reference][prefix+'Full'] for r in rows])
  note=np.array([r[model][prefix+'Note']-r[reference][prefix+'Note'] for r in rows])
  check(float(full.mean()),want['Full'],(pair,condition,'Full'))
  check(float(note.mean()),want['Note'],(pair,condition,'Note'))
  assert int((full>0).sum())==want['Full_up'],(pair,condition)
  assert int(((full>0)&(note<0)).sum())==want['Full_up_Note_down'],(pair,condition)
print(json.dumps(dict(independent_cpu_replay=True,tracks=len(rows),predictors=len(models),values_checked=checked,max_absolute_difference=max_error,all_pair_counts_match=True),indent=2))
