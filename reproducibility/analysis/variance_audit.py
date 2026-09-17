from pathlib import Path
import json,hashlib
import numpy as np
P=Path(__file__).resolve().parent;C=P if (P/'analysis_arrays.npz').exists() else P/'companion';data=np.load(C/'analysis_arrays.npz');tracks=json.loads((C/'tracks.json').read_text());old=json.loads((P/'diagnostic_audit.json' if (P/'diagnostic_audit.json').exists() else P/'expected_results.json').read_text());models=list(old['per_track'])
thresholds=[1e-10,1e-8,1e-6,1e-4];result={str(x):{m:[] for m in models} for x in thresholds};sds=[];counts=[];boundaries=[]
def corr(a,b):
    if len(a)<3 or np.std(a)<1e-10 or np.std(b)<1e-10:return None
    a=a-a.mean();b=b-b.mean();return float(np.dot(a,b)/np.sqrt(np.dot(a,a)*np.dot(b,b)))
for r in tracks:
    pre=r['prefix'];y=data[pre+'_amp'].astype(float);notes=data[pre+'_notes'];h=r['hop_time']
    spans=[(max(0,int(float(n[0])/h)),min(len(y),int(float(n[1])/h))) for n in notes];spans=[(a,b) for a,b in spans if b-a>=4]
    target_sd=np.array([np.std(y[a:b]) for a,b in spans]);sds.extend(target_sd.tolist());defined={}
    boundaries.append({'track':r['track'],'dataset':r['dataset'],'notes':len(notes),'adjacent_end_equals_next_start':int(np.sum(notes[:-1,1]==notes[1:,0])),'adjacent_pairs':len(notes)-1})
    for m in models:
        pred=data[pre+'_'+m].astype(float);cr=[corr(pred[a:b],y[a:b]) for a,b in spans];defined[m]=sum(v is not None for v in cr)
        for threshold in thresholds:
            selected=[v for v,sd in zip(cr,target_sd) if v is not None and sd>=threshold]
            result[str(threshold)][m].append({'track':r['track'],'retained_notes':int(np.sum(target_sd>=threshold)),'defined_notes':len(selected),'within_note':float(np.mean(selected)) if selected else None})
    counts.append({'track':r['track'],'eligible_notes':len(spans),'defined_notes':defined})
summary={}
for threshold,values in result.items():
    summary[threshold]={}
    for m,rs in values.items():
        v=[r['within_note'] for r in rs if r['within_note'] is not None]
        summary[threshold][m]={'mean':float(np.mean(v)) if v else None,'tracks':len(v),'retained_notes':sum(r['retained_notes'] for r in rs),'defined_notes':sum(r['defined_notes'] for r in rs)}
for m in models:
    assert abs(summary['1e-10'][m]['mean']-old['subsets']['all186']['summary'][m]['within_note']['mean'])<1e-10 if summary['1e-10'][m]['mean'] is not None else old['subsets']['all186']['summary'][m]['within_note']['mean'] is None
corpora={}
for corpus in sorted({r['dataset'] for r in tracks}):
    ts=[r['track'] for r in tracks if r['dataset']==corpus];gs={r['track']:r['group'] for r in tracks};names=sorted({gs[t] for t in ts});lookup={m:{r['track']:r['views']['all'] for r in old['per_track'][m]} for m in models}
    d=np.array([lookup['mamba'][t]['note_mean']-lookup['instrument_ols'][t]['note_mean'] for t in ts]);group_ids=[names.index(gs[t]) for t in ts]
    draws=np.random.RandomState(20260911).randint(len(names),size=(5000,len(names)));gw=np.stack([(draws==j).sum(1) for j in range(len(names))],axis=1);w=gw[:,group_ids];v=w@d/w.sum(1)
    corpora[corpus]={'tracks':len(ts),'groups':len(names),'mamba_minus_instrument_ols_note_mean':float(d.mean()),'ci95':np.quantile(v,[.025,.975]).tolist(),'boundaries_equal':sum(r['adjacent_end_equals_next_start'] for r in boundaries if r['dataset']==corpus),'adjacent_pairs':sum(r['adjacent_pairs'] for r in boundaries if r['dataset']==corpus)}
report={'source_npz_sha256':hashlib.sha256((C/'analysis_arrays.npz').read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256((P/'ROBUSTNESS_PLAN.md').read_bytes()).hexdigest(),'target_std_quantiles':dict(zip(['min','p01','p05','p50','p95','max'],np.quantile(sds,[0,.01,.05,.5,.95,1]).tolist())),'low_std_note_counts':{str(t):int(np.sum(np.array(sds)<t)) for t in thresholds},'eligible_notes':len(sds),'eligible_notes_per_track_quantiles':np.quantile([r['eligible_notes'] for r in counts],[0,.25,.5,.75,1]).tolist(),'defined_notes_per_track_quantiles':{m:np.quantile([r['defined_notes'][m] for r in counts],[0,.25,.5,.75,1]).tolist() for m in models},'summaries':summary,'per_track':result,'note_counts':counts,'annotation_boundaries':boundaries,'corpora':corpora}
(P/'variance_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ('per_track','note_counts','annotation_boundaries')},indent=2))
