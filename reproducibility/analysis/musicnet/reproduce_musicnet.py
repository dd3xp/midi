"""CPU replay of every saved transfer metric and work-group contrast; no fitting."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.ndimage import gaussian_filter1d
from metrics_core import metrics,references

P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((P/'manifest.json').read_text())
for rel,h in manifest.items():assert sha(P/rel)==h,rel
tracks=json.loads((P/'tracks.json').read_text());a=np.load(P/'inputs.npz');pred=np.load(P/'predictions_all.npz')
r=json.loads((P/'results.json').read_text());selection=json.loads((P/'selection.json').read_text())
coef=json.loads((P/'coefficients.json').read_text())
assert len(tracks)==len({t['track'] for t in tracks})==21
assert len({t['group'] for t in tracks})==5
models=sorted(r['summary']);keys=['Full','Active','Note','Within','Error','b2','c2']
lookup={(v['model'],v['track']):v for v in r['per_track']}
assert len(lookup)==len(r['per_track'])==147
maximum=0.;actual={}
def check(x,y,tol=1e-10):
    global maximum
    assert (x is None)==(y is None),(x,y)
    if x is not None:
        delta=float(np.max(np.abs(np.asarray(x)-np.asarray(y))))
        maximum=max(maximum,delta);assert delta<tol,(x,y,delta)
for t in tracks:
    prefix=t['track'].split('/')[-1]
    row=dict(t,**{k:a[prefix+'_'+k] for k in ['amp','notes','features']})
    y=row['amp'].astype(float)
    assert len(y)==t['frames'] and np.isfinite(y).all() and (y>=0).all()
    refs,clipped=references(row,coef)
    for m in models:
        p=pred[prefix+'_'+m].astype(float)
        assert p.shape==y.shape and np.isfinite(p).all() and (p>=0).all()
        if m in refs:assert np.array_equal(p,refs[m]),(m,prefix)
        s=metrics(p,y,row['notes'],row['hop_time']);original=lookup[m,t['track']]
        for k,val in s.items():check(val,original[k])
        assert original['clipped']==clipped.get(m,0)
        views={'log_corr':float(np.corrcoef(np.log(p+1e-7),np.log(y+1e-7))[0,1])}
        for sigma in [.05,.1,.2,.5,1]:
            for mode in ['log','linear']:
                x,z=(np.log(p+1e-7),np.log(y+1e-7)) if mode=='log' else (p,y)
                views[f'{mode}_smooth_{sigma:g}s']=float(np.corrcoef(gaussian_filter1d(x,sigma/t['hop_time']),gaussian_filter1d(z,sigma/t['hop_time']))[0,1])
        for k,val in views.items():check(val,original['fixed_output_scales'][k])
        actual[m,t['track']]=s
groups=r['groups'];draws=np.random.default_rng(20260916).integers(0,5,size=(5000,5))
for m in models:
    for k in keys:
        vals=[actual[m,t['track']][k] for t in tracks if actual[m,t['track']][k] is not None]
        assert len(vals)==r['summary'][m][k]['n']
        check(float(np.mean(vals)) if vals else None,r['summary'][m][k]['mean'])
        means=[]
        for g in groups:
            x=[actual[m,t['track']][k] for t in tracks if t['group']==g and actual[m,t['track']][k] is not None]
            v=float(np.mean(x)) if x else None
            check(v,r['work_summary'][m][g][k])
            if v is not None:means.append(v)
        check(float(np.mean(means)) if means else None,r['equal_work_summary'][m][k])
for name,pair in r['pairs'].items():
    model,ref=name.split(' minus ')
    for k in keys:
        vs=[[actual[model,t['track']][k]-actual[ref,t['track']][k] for t in tracks if t['group']==g and actual[model,t['track']][k] is not None and actual[ref,t['track']][k] is not None] for g in groups]
        flat=[v for x in vs for v in x];d=pair['deltas'][k]
        assert d['n']==len(flat)
        check(float(np.mean(flat)),d['mean'])
        check(float(np.mean([np.mean(x) for x in vs if x])),d['equal_work_mean'])
        for g,x in zip(groups,vs):check(float(np.mean(x)) if x else None,pair['by_work'][k][g])
        boot=[float(np.mean([v for i in dr for v in vs[i]])) for dr in draws if any(vs[i] for i in dr)]
        check(np.percentile(boot,[2.5,97.5]).tolist(),d['exploratory_group_ci'])
    up=[t for t in tracks if actual[model,t['track']]['Full']>actual[ref,t['track']]['Full']]
    down=[t['track'] for t in up if actual[model,t['track']]['Note']<actual[ref,t['track']]['Note']]
    assert len(up)==pair['full_improved'] and down==pair['discordant_tracks'] and len(down)==pair['note_decreased']
for k in ['Full','Note']:
    means={m:float(np.mean([x[k] for x in selection['per_track'] if x['model']==m])) for m in selection['eligible']}
    for m,v in means.items():check(v,selection['summary'][m][k]['mean'])
    winner=min(m for m,v in means.items() if max(means.values())-v<=1e-12)
    assert winner==selection['selected'][k]==r['selection'][k]
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'passed','recordings':21,'works':5,'models':7,'pairs':6,'all_metrics_and_fixed_scales_recomputed':True,'all_work_and_bootstrap_results_recomputed':True,'reference_outputs_reconstructed_without_fitting':True,'validation_choice_recomputed_from_saved_validation_metrics':True,'maximum_difference':maximum,'input_manifest_checked':True}
(P/'recomputed_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
