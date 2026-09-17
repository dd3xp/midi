"""Recompute paired corpus diagnostics from retained arrays, no fitting."""
from pathlib import Path
from datetime import datetime,timezone
import numpy as np, json, hashlib
P=Path(__file__).resolve().parent
C=P.parent/'revision_20260915/companion'
if (P.parent/'analysis_arrays.npz').exists():C=P.parent
R=P.parent/'refinement_20260915'
if not R.exists():R=P.parent/'refinement'
meta=json.loads((C/'tracks.json').read_text());a=np.load(C/'analysis_arrays.npz')
models=['mamba','s4d','bigru','instrument_ols','note_shape']
stored=json.loads((C/'expected_results.json').read_text())['per_track']
lookup={m:{x['track']:x['views']['all'] for x in v} for m,v in stored.items()}
errs=json.loads((R/'log_error_parts.json').read_text())['per_track']
elook={m:{x['track']:x for x in v} for m,v in errs.items()}
def corr(x,y):
    x=x-x.mean();y=y-y.mean()
    if len(x)<3 or np.std(x)<1e-10 or np.std(y)<1e-10:return None
    return float(x@y/np.sqrt((x@x)*(y@y)))
per={};maxerr=0.
for m in models:
    out=[]
    for t in meta:
        pre=t['prefix'];y=a[pre+'_amp'].astype(float);p=a[pre+'_'+m].astype(float);T=len(y);h=t['hop_time']
        spans=[(max(0,int(float(n[0])/h)),min(T,int(float(n[1])/h))) for n in a[pre+'_notes']]
        spans=[(s,e) for s,e in spans if e-s>=4]
        # Explicit means here independently check the existing prefix-sum error audit.
        ym=np.array([y[s:e].mean() for s,e in spans]);pm=np.array([p[s:e].mean() for s,e in spans])
        d=np.log(pm+1e-7)-np.log(ym+1e-7);b=d.mean();c2=((d-b)**2).mean();E2=(d*d).mean()
        vals=dict(Full=corr(p,y),Note=corr(pm,ym),Error=np.sqrt(E2),b2=b*b,c2=c2,E2=E2)
        exp=lookup[m][t['track']];ee=elook[m][t['track']]
        for x,z in [(vals['Full'],exp['raw']),(vals['Note'],exp['note_mean']),(vals['Error'],ee['E']),(b*b,ee['b2']),(c2,ee['c2'])]:
            maxerr=max(maxerr,abs(x-z));assert abs(x-z)<1e-8
        assert abs(E2-b*b-c2)<1e-10
        out.append(dict(track=t['track'],**vals))
    per[m]=out
populations={name:[i for i,t in enumerate(meta) if t['dataset']==name] for name in sorted({t['dataset'] for t in meta})}
populations['nonURMP']=[i for i,t in enumerate(meta) if t['dataset']!='URMP'];populations['all']=list(range(186))
result={}
for name,idx in populations.items():
    groups=sorted({meta[i]['group'] for i in idx});n=len(idx);ng=len(groups)
    draws=np.random.RandomState(20260911).randint(ng,size=(5000,ng))
    wg=np.stack([(draws==g).sum(1) for g in range(ng)],axis=1)
    w=wg[:,[groups.index(meta[i]['group']) for i in idx]]
    def summary(x):
        return dict(mean=float(x.mean()),ci95=np.quantile(w@x/w.sum(1),[.025,.975]).tolist())
    comp={}
    for m in models[:3]:
        for ref in models[3:]:
            dif={k:np.array([per[m][i][k]-per[ref][i][k] for i in idx]) for k in ('Full','Note','Error','b2','c2','E2')}
            up=dif['Full']>0;down=up&(dif['Note']<0);bad=up&(dif['Error']>0)
            cd=dict(full_up=int(up.sum()),note_down=int(down.sum()),error_up=int(bad.sum()),
                note_loss_median=float(np.median(-dif['Note'][down])) if down.any() else None,
                paired={k:summary(v) for k,v in dif.items()})
            comp[m+' minus '+ref]=cd
    result[name]=dict(tracks=n,groups=ng,comparisons=comp,
        component_means={m:{k:float(np.mean([per[m][i][k] for i in idx])) for k in ('b2','c2','E2')} for m in models})
frozen=json.loads((R/'all_predictor_results.json').read_text())['comparisons']
for key,v in result['all']['comparisons'].items():
    for k,old in [('full_up','full_improved'),('note_down','note_decreased'),('error_up','error_increased')]:assert v[k]==frozen[key][old]
report=dict(utc=datetime.now(timezone.utc).isoformat(),populations=result,per_track=per,
    verification=dict(maximum_difference=maxerr,checked_values=186*5*5,all_population_counts_match=True),
    sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (C/'analysis_arrays.npz',C/'tracks.json',P/'PLAN.md',Path(__file__))})
(P/'corpus_results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
for name,d in result.items():
    print(name,d['tracks'],d['groups'],{k:(v['note_down'],v['full_up'],round(v['paired']['Note']['mean'],4)) for k,v in d['comparisons'].items()})
print('verification',report['verification'])
