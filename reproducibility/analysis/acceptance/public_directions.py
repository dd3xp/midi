from pathlib import Path
from datetime import datetime,timezone
import numpy as np,json,hashlib
P=Path(__file__).resolve().parent;C=P.parent/'revision_20260915/companion'
if (P.parent/'analysis_arrays.npz').exists():C=P.parent
meta=json.loads((C/'tracks.json').read_text());tracks={x['track']:x for x in meta}
manifest=json.loads((C/'public_rms_manifest.json').read_text())
expected=json.loads((C/'public_expected_results.json').read_text())
a=np.load(C/'analysis_arrays.npz');pub=np.load(C/'public_rms_arrays.npz')
def corr(x,y):return float(np.corrcoef(x,y)[0,1])
def metric(x,t):
    pre=t['prefix'];y=a[pre+'_amp'].astype(float);x=x.astype(float);h=t['hop_time'];assert np.isfinite(x).all() and len(x)==len(y)
    spans=[(max(0,int(float(n[0])/h)),min(len(y),int(float(n[1])/h))) for n in a[pre+'_notes']]
    spans=[(s,e) for s,e in spans if e-s>=4]
    xm=np.array([x[s:e].mean() for s,e in spans]);ym=np.array([y[s:e].mean() for s,e in spans])
    return dict(Full=corr(x,y),Note=corr(xm,ym),Error=float(np.sqrt(np.mean((np.log(xm+1e-7)-np.log(ym+1e-7))**2))))
systems={}
for model,info in manifest['systems'].items():
    per={};exp={x['track']:x['metrics'] for x in expected[model]['per_track']}
    for row in info['rows']:
        t=tracks[row['track']];metrics=metric(pub[row['array_key']],t)
        assert abs(metrics['Full']-exp[t['track']]['raw'])<1e-8 and abs(metrics['Note']-exp[t['track']]['note_mean'])<1e-8
        per[t['track']]={model:metrics,**{ref:metric(a[t['prefix']+'_'+ref],t) for ref in ['instrument_ols','note_shape']}}
    subsets={}
    for label,sub in expected[model]['subsets'].items():
        ts=sub['tracks'];groups=sorted({tracks[t]['group'] for t in ts});ng=len(groups)
        draws=np.random.RandomState(20260911).randint(ng,size=(5000,ng));wg=np.stack([(draws==g).sum(1) for g in range(ng)],axis=1);w=wg[:,[groups.index(tracks[t]['group']) for t in ts]]
        comparisons={}
        for ref in ['instrument_ols','note_shape']:
            d={k:np.array([per[t][model][k]-per[t][ref][k] for t in ts]) for k in ['Full','Note','Error']}
            up=d['Full']>0;down=up&(d['Note']<0);err=up&(d['Error']>0)
            comparisons[ref]=dict(full_up=int(up.sum()),note_down=int(down.sum()),error_up=int(err.sum()),
                note_loss_median=float(np.median(-d['Note'][down])) if down.any() else None,
                paired={k:dict(mean=float(v.mean()),ci95=np.quantile(w@v/w.sum(1),[.025,.975]).tolist()) for k,v in d.items()})
        subsets[label]=dict(tracks=len(ts),groups=ng,comparisons=comparisons)
    systems[model]=dict(subsets=subsets,per_track=per)
out=dict(utc=datetime.now(timezone.utc).isoformat(),systems=systems,scope='Fixed adapted pipeline output diagnostics, not fair architecture ranking or independent unseen-data evaluation',sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [C/'analysis_arrays.npz',C/'public_rms_arrays.npz',P/'PLAN.md',Path(__file__)]})
(P/'public_directions.json').write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({m:d['subsets'] for m,d in systems.items()},indent=2))
