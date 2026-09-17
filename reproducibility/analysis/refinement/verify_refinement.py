"""Independent interval construction, moment correlations and group resampling."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
P=Path(__file__).resolve().parent
C=P.parent if (P.parent/'analysis_arrays.npz').exists() else P.parent/'revision_20260915/companion'
r=json.loads((P/'results.json').read_text());rows=json.loads((C/'tracks.json').read_text());a=np.load(C/'analysis_arrays.npz')
maxerr=0.;checks=0
def close(x,y):
    global maxerr,checks
    if x is None or y is None:assert x is None and y is None;return
    d=float(np.max(np.abs(np.asarray(x)-y)));maxerr=max(maxerr,d);checks+=1;assert d<1e-8,(x,y,d)
def pc(x,y):
    x=x-x.mean();y=y-y.mean()
    if len(x)<3 or np.sqrt(np.mean(x*x))<1e-10 or np.sqrt(np.mean(y*y))<1e-10:return None
    return float(x@y/np.sqrt((x@x)*(y@y)))
models=list(r['boundary_per_track'][0]['original'])
for i,row in enumerate(rows):
    pre=row['prefix'];notes=a[pre+'_notes'];y=a[pre+'_amp'].astype(float);h=row['hop_time'];T=len(y)
    starts=np.maximum(0,(notes[:,0].astype(float)/h).astype(int));ends=np.minimum(T,(notes[:,1].astype(float)/h).astype(int))
    candidates=sorted(set(float(n) for n in notes[:,0])); successor=dict(zip(candidates[:-1],candidates[1:]))
    alternative=np.minimum(T,np.array([int(successor.get(float(n[0]),float(n[1]))/h) for n in notes]))
    keep=(ends-starts>=4)&(alternative-starts>=4);s=starts[keep];e0=ends[keep];e1=alternative[keep]
    assert len(s)==r['boundary_per_track'][i]['common_eligible']
    for rule,e in [('original',e0),('next',e1)]:
        yc=np.r_[0.,y.cumsum()];ym=(yc[e]-yc[s])/(e-s)
        for m in models:
            p=a[pre+'_'+m].astype(float);cp=np.r_[0.,p.cumsum()];pm=(cp[e]-cp[s])/(e-s)
            expected=r['boundary_per_track'][i][rule][m]
            close(pc(pm,ym),expected['note_mean'])
            close(float(np.sqrt(np.mean((np.log(pm+1e-7)-np.log(ym+1e-7))**2))),expected['note_log_rmse'])
    for ref in r['directions']:
        new=r['directions'][ref]['per_track'][i]
        close(pc(a[pre+'_mamba'].astype(float),y)-pc(a[pre+'_'+ref].astype(float),y),new['full_gain'])
        close(r['boundary_per_track'][i]['original']['mamba']['note_mean']-r['boundary_per_track'][i]['original'][ref]['note_mean'],new['note_gain'])
def boot(v,ids):
    gs=sorted({rows[i]['group'] for i in ids})
    sums=np.array([sum(v[j] for j,i in enumerate(ids) if rows[i]['group']==g) for g in gs])
    counts=np.array([sum(rows[i]['group']==g for i in ids) for g in gs])
    ix=np.random.RandomState(20260911).choice(len(gs),size=(5000,len(gs)),replace=True)
    return np.quantile(sums[ix].sum(1)/counts[ix].sum(1),[.025,.975])
subsets={'all186':list(range(186)),'URMP136':[i for i,x in enumerate(rows) if x['dataset']=='URMP'],
 'nonURMP50':[i for i,x in enumerate(rows) if x['dataset']!='URMP'],'nonBach146':[i for i,x in enumerate(rows) if x['dataset']!='Bach10']}
for subset,ids in subsets.items():
    for ref,d in r['boundary_summary'][subset].items():
        for metric,rs in d.items():
            vals={rule:np.array([r['boundary_per_track'][i][rule]['mamba'][metric]-r['boundary_per_track'][i][rule][ref][metric] for i in ids]) for rule in ['original','next']}
            vals['rule_change']=vals['next']-vals['original']
            for rule,v in vals.items():close(v.mean(),rs[rule]['mean']);close(boot(v,ids),rs[rule]['ci95'])
for name,d in r['directions'].items():
    full=np.array([x['full_gain'] for x in d['per_track']]);note=np.array([x['note_gain'] for x in d['per_track']])
    close(int(((full>0)&(note<0)).sum()),d['counts']['full_1_note_-1'])
    gs=sorted({x['group'] for x in rows});iv=np.random.RandomState(20260911).choice(len(gs),size=(5000,len(gs)),replace=True)
    nums=np.array([sum(full[i]>0 and note[i]<0 for i,x in enumerate(rows) if x['group']==g) for g in gs]);dens=np.array([sum(full[i]>0 for i,x in enumerate(rows) if x['group']==g) for g in gs])
    close(np.quantile(nums[iv].sum(1)/dens[iv].sum(1),[.025,.975]),d['discordance_given_full_up']['ci95'])
for name,h in r['sha256'].items():
    f=P/name if (P/name).exists() else C/name
    assert hashlib.sha256(f.read_bytes()).hexdigest()==h,name
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'passed','numeric_checks':checks,'maximum_error':maxerr,'results_sha256':hashlib.sha256((P/'results.json').read_bytes()).hexdigest()}
(P/'verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
