from pathlib import Path
import datetime,hashlib,json
import numpy as np
from scipy.ndimage import gaussian_filter1d
P=Path(__file__).resolve().parent;C=P.parent if (P.parent/'analysis_arrays.npz').exists() else P.parent/'revision_20260915/companion'
r=json.loads((P/'results.json').read_text());tracks=json.loads((C/'tracks.json').read_text());a=np.load(C/'analysis_arrays.npz')
old=json.loads((C/'expected_results.json').read_text());models=list(old['per_track'])
def corr(x,y):
    x=x-x.mean();y=y-y.mean()
    if len(y)<3 or np.std(x)<1e-10 or np.std(y)<1e-10:return None
    return float(x@y/np.sqrt((x@x)*(y@y)))
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'magnitude':{},'scale_per_track':[],
 'sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [P/'SCALE_AND_MAGNITUDE_PLAN.md',P/'scale_and_magnitude.py',P/'results.json',C/'analysis_arrays.npz']}}
for ref,d in r['directions'].items():
    dx=np.array([x['full_gain'] for x in d['per_track']]);loss=-np.array([x['note_gain'] for x in d['per_track']]);ix=(dx>0)&(loss>0)
    out['magnitude'][ref]={'full_gain_quartiles':np.quantile(dx[ix],[.25,.5,.75]).tolist(),'note_loss_quartiles':np.quantile(loss[ix],[.25,.5,.75]).tolist(),
        'thresholds':[{'cutoff':t,'both_exceed':int(((dx>t)&(loss>t)).sum()),'full_exceeds':int((dx>t).sum()),'all_tracks':len(dx)} for t in [0,.01,.025,.05,.1]]}
maxerr=0.
conditions=[('linear',None,s) for s in [0,.05,.1,.2,.5,1]]+[('log',eps,s) for eps in [1e-8,1e-7,1e-6] for s in [0,.05,.1,.2,.5,1]]
def key(d,e,s):return f'{d}_eps{e}_sigma{s}'
for i,row in enumerate(tracks):
    pre=row['prefix'];y=a[pre+'_amp'].astype(float);T=len(y);h=row['hop_time'];mask=np.zeros(T,bool)
    for n in a[pre+'_notes']:
        s=max(0,int(float(n[0])/h));e=min(T,int(float(n[1])/h));mask[s:e]=True
    z=np.stack([y]+[a[pre+'_'+m].astype(float) for m in models])
    rr={'index':i,'track':row['track'],'conditions':{}}
    for dom,eps,sig in conditions:
        q=z if dom=='linear' else np.log(z+eps)
        if sig:q=gaussian_filter1d(q,sigma=sig/h,axis=1,mode='reflect',truncate=4.)
        rs={}
        for j,m in enumerate(models,1):
            rs[m]={'full':corr(q[j],q[0]),'active':corr(q[j,mask],q[0,mask])}
            if dom=='linear' and sig==0:
                prior=old['per_track'][m][i];assert prior['track']==row['track']
                for v,oldv in [('full','raw'),('active','active_raw')]:
                    expected=prior['views']['all'][oldv]
                    if expected is None:assert rs[m][v] is None
                    else:d=abs(expected-rs[m][v]);maxerr=max(maxerr,d);assert d<1e-8
        rr['conditions'][key(dom,eps,sig)]=rs
    out['scale_per_track'].append(rr)
    if i%30==0:print('scale tracks',i,flush=True)
out['scale_summary']={}
for dom,eps,sig in conditions:
    k=key(dom,eps,sig);out['scale_summary'][k]={}
    for m in models:
        out['scale_summary'][k][m]={}
        for view in ['full','active']:
            vs=[row['conditions'][k][m][view] for row in out['scale_per_track']];defined=[v for v in vs if v is not None]
            out['scale_summary'][k][m][view]={'mean':float(np.mean(defined)) if defined else None,'tracks':len(defined)}
out['primary_linear_max_error']=maxerr
out['epsilon_sensitivity']={}
for m in models:
    out['epsilon_sensitivity'][m]={}
    for view in ['full','active']:
        changes=[]
        for eps in [1e-8,1e-6]:
            for s in [0,.05,.1,.2,.5,1]:
                x=out['scale_summary'][key('log',eps,s)][m][view]['mean'];y=out['scale_summary'][key('log',1e-7,s)][m][view]['mean']
                if x is not None and y is not None:changes.append(abs(x-y))
        out['epsilon_sensitivity'][m][view]=max(changes) if changes else None
(P/'scale_results.json').write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({'magnitude':out['magnitude'],'epsilon':out['epsilon_sensitivity'],
 'mamba_active':{k:v['mamba']['active'] for k,v in out['scale_summary'].items() if 'eps1e-07' in k or k.startswith('linear')},
 'maxerr':maxerr},indent=2))
