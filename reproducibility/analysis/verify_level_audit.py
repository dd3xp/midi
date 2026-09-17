"""Independent check: cumulative-sum interval means and explicit group resampling.

Does not modify arrays or the expected report. Also checks alternative aggregation
using concatenated bootstrap rows rather than the audit's weight matrices.
"""
from pathlib import Path
import json, hashlib
import numpy as np
P=Path(__file__).resolve().parent
C=P if (P/'analysis_arrays.npz').exists() else P/'companion'
reference=json.loads((C/'level_and_aggregation_audit.json').read_text(encoding='utf-8'))
original=json.loads((C/'expected_results.json').read_text(encoding='utf-8'))
tracks=json.loads((C/'tracks.json').read_text(encoding='utf-8'))
arrays=np.load(C/'analysis_arrays.npz')
errors={m:{} for m in reference['error_per_track']}
maximum=0.;metric_checks=0;paired_checks=0
def check(a,b):
    global maximum
    delta=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    maximum=max(maximum,delta)
    assert delta<1e-8,delta
for row in tracks:
    pre=row['prefix'];y=arrays[pre+'_amp'].astype(np.float64)
    edges=np.trunc(arrays[pre+'_notes'][:,:2].astype(np.float64)/row['hop_time']).astype(np.int64)
    starts=np.maximum(0,edges[:,0]);ends=np.minimum(len(y),edges[:,1])
    selected=(ends-starts)>=4;starts=starts[selected];ends=ends[selected]
    def interval_means(a):
        prefix=np.r_[0.,np.cumsum(a,dtype=np.float64)]
        return (prefix[ends]-prefix[starts])/(ends-starts)
    target=interval_means(y)
    for model in errors:
        means=interval_means(arrays[pre+'_'+model]);d=means-target
        logd=np.log(means+1e-7)-np.log(target+1e-7)
        values={'note_linear_rmse':float(np.linalg.norm(d)/np.sqrt(len(d))),
                'note_log_rmse':float(np.linalg.norm(logd)/np.sqrt(len(d))),
                'note_linear_mae':float(np.sum(np.abs(d))/len(d))}
        errors[model][row['track']]=values
        for key,value in values.items():
            check(value,reference['error_per_track'][model][row['track']][key]);metric_checks+=1
groups={r['track']:r['group'] for r in tracks}
names=[r['track'] for r in tracks]
def compare(x,y,ids,kind,expected):
    global paired_checks
    ordered=sorted({groups[t] for t in ids})
    rows=[np.array([i for i,t in enumerate(ids) if groups[t]==g]) for g in ordered]
    draws=np.random.RandomState(20260911).randint(len(rows),size=(5000,len(rows)))
    if kind=='group_arithmetic':
        left=np.array([x[idx].mean() for idx in rows]);right=np.array([y[idx].mean() for idx in rows])
        point=left.mean()-right.mean();samples=(left[draws]-right[draws]).mean(axis=1)
    else:
        if kind=='track_fisher':
            x=np.arctanh(np.clip(x,-1+1e-7,1-1e-7));y=np.arctanh(np.clip(y,-1+1e-7,1-1e-7));transform=np.tanh
        else:transform=lambda a:a
        point=transform(x.mean())-transform(y.mean())
        samples=np.empty(5000)
        for j,draw in enumerate(draws):
            idx=np.concatenate([rows[k] for k in draw])
            samples[j]=transform(x[idx].mean())-transform(y[idx].mean())
    check(point,expected['difference']);check(np.percentile(samples,[2.5,97.5]),expected['ci95']);paired_checks+=1
for pair,metrics in reference['paired_note_errors'].items():
    a,b=pair.split(' minus ')
    for key,value in metrics.items():
        compare(np.array([errors[a][t][key] for t in names]),np.array([errors[b][t][key] for t in names]),names,'track_arithmetic',value)
lookup={m:{r['track']:r['views']['all'] for r in vals} for m,vals in original['per_track'].items()}
for pair,metrics in reference['aggregation_checks'].items():
    a,b=pair.split(' minus ')
    for key,methods in metrics.items():
        ids=[t for t in names if lookup[a][t][key] is not None and lookup[b][t][key] is not None]
        x=np.array([lookup[a][t][key] for t in ids]);y=np.array([lookup[b][t][key] for t in ids])
        for kind,value in methods.items():compare(x,y,ids,kind,value)
result={'status':'passed','note_error_metric_checks':metric_checks,'paired_error_and_aggregation_checks':paired_checks,'max_error':maximum,
        'source_npz_sha256':hashlib.sha256((C/'analysis_arrays.npz').read_bytes()).hexdigest(),
        'implementation':'cumulative sums and explicit resampled group-row concatenation; no refitting'}
(P/'level_independent_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
