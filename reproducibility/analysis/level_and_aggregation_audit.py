from pathlib import Path
import json,hashlib
import numpy as np
P=Path(__file__).resolve().parent;C=P if (P/'analysis_arrays.npz').exists() else P/'companion'
r=json.loads((C/'expected_results.json').read_text());tracks=json.loads((C/'tracks.json').read_text());a=np.load(C/'analysis_arrays.npz');models=list(r['per_track']);ts=[x['track'] for x in tracks];groups={x['track']:x['group'] for x in tracks};pairs=[('mamba','instrument_ols'),('mamba','note_shape'),('mamba','bigru')]
errors={m:{} for m in models}
for row in tracks:
    pre=row['prefix'];y=a[pre+'_amp'].astype(float);notes=a[pre+'_notes'];h=row['hop_time'];spans=[(max(0,int(float(n[0])/h)),min(len(y),int(float(n[1])/h))) for n in notes];spans=[(s,e) for s,e in spans if e-s>=4]
    ym=np.array([y[s:e].mean() for s,e in spans])
    for m in models:
        pred=a[pre+'_'+m].astype(float);pm=np.array([pred[s:e].mean() for s,e in spans]);d=pm-ym;dl=np.log(pm+1e-7)-np.log(ym+1e-7)
        errors[m][row['track']]={'note_linear_rmse':float(np.sqrt(np.mean(d*d))),'note_log_rmse':float(np.sqrt(np.mean(dl*dl))),'note_linear_mae':float(np.mean(np.abs(d)))}
summary={m:{k:float(np.mean([errors[m][t][k] for t in ts])) for k in next(iter(errors[m].values()))} for m in models}
def paired(x,y,tracklist,kind):
    gn=sorted({groups[t] for t in tracklist});ix=[gn.index(groups[t]) for t in tracklist];counts=np.bincount(ix);draws=np.random.RandomState(20260911).randint(len(gn),size=(5000,len(gn)));gw=np.stack([(draws==j).sum(1) for j in range(len(gn))],axis=1);w=gw[:,ix]
    if kind=='group_arithmetic':
        gx=np.bincount(ix,weights=x)/counts;gy=np.bincount(ix,weights=y)/counts;dx=gx.mean()-gy.mean();bs=(gw@(gx-gy))/gw.sum(1)
    elif kind=='track_fisher':
        zx=np.arctanh(np.clip(x,-1+1e-7,1-1e-7));zy=np.arctanh(np.clip(y,-1+1e-7,1-1e-7));dx=np.tanh(zx.mean())-np.tanh(zy.mean());bs=np.tanh(w@zx/w.sum(1))-np.tanh(w@zy/w.sum(1))
    else:dx=x.mean()-y.mean();bs=w@(x-y)/w.sum(1)
    return {'difference':float(dx),'ci95':np.quantile(bs,[.025,.975]).tolist(),'tracks':len(tracklist),'groups':len(gn)}
ep={}
for m,n in pairs:
    ep[m+' minus '+n]={k:paired(np.array([errors[m][t][k] for t in ts]),np.array([errors[n][t][k] for t in ts]),ts,'track_arithmetic') for k in summary[m]}
lookup={m:{v['track']:v['views']['all'] for v in r['per_track'][m]} for m in models};agg={}
for m,n in pairs:
    agg[m+' minus '+n]={}
    for metric in ('raw','active_raw','note_mean','within_note'):
        tl=[t for t in ts if lookup[m][t][metric] is not None and lookup[n][t][metric] is not None];x=np.array([lookup[m][t][metric] for t in tl]);y=np.array([lookup[n][t][metric] for t in tl]);agg[m+' minus '+n][metric]={kind:paired(x,y,tl,kind) for kind in ('track_arithmetic','group_arithmetic','track_fisher')}
        q=agg[m+' minus '+n][metric]['track_arithmetic'];old=r['subsets']['all186']['paired'][m+' minus '+n][metric];assert abs(q['difference']-old['difference'])<1e-10 and np.max(np.abs(np.array(q['ci95'])-old['ci95']))<1e-10
out={'source_npz_sha256':hashlib.sha256((C/'analysis_arrays.npz').read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256((P/'LEVEL_AND_AGGREGATION_PLAN.md').read_bytes()).hexdigest(),'error_summary':summary,'paired_note_errors':ep,'error_per_track':errors,'aggregation_checks':agg}
(P/'level_and_aggregation_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in out.items() if k!='error_per_track'},indent=2))
