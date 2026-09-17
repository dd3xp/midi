"""Recompute public-pipeline claims from the archived final-audio RMS arrays.

Uses NumPy and SciPy; no audio rendering, network, GPU or external file paths.
"""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.ndimage import gaussian_filter1d
P=Path(__file__).resolve().parent
public=np.load(P/'public_rms_arrays.npz');internal=np.load(P/'analysis_arrays.npz')
meta=json.loads((P/'public_rms_manifest.json').read_text(encoding='utf-8'))
reports=json.loads((P/'public_expected_results.json').read_text(encoding='utf-8'))
tracks={x['track']:x for x in json.loads((P/'tracks.json').read_text(encoding='utf-8'))}
prior=json.loads((P/'expected_results.json').read_text(encoding='utf-8'))
assert hashlib.sha256((P/'public_rms_arrays.npz').read_bytes()).hexdigest()==meta['npz_sha256']
keys=('raw','active_raw','note_mean','within_note','log_corr','log_smooth_1s')
calculated={};maximum=0.;metric_checks=summary_checks=pair_checks=0
def corr(x,y):
    if len(x)<3 or np.std(x)<1e-10 or np.std(y)<1e-10:return None
    x=x-x.mean();y=y-y.mean();return float(np.dot(x,y)/np.sqrt(np.dot(x,x)*np.dot(y,y)))
def check(x,y):
    global maximum
    if x is None or y is None:assert x is None and y is None;return
    e=float(np.max(np.abs(np.asarray(x)-np.asarray(y))));maximum=max(maximum,e);assert e<1e-8,e
for model,info in meta['systems'].items():
    stored={r['track']:r for r in reports[model]['per_track']};calculated[model]={}
    assert len(info['rows'])==len(stored) and len({r['track'] for r in info['rows']})==len(stored)
    for row in info['rows']:
        t=row['track'];r=tracks[t];pre=r['prefix'];h=r['hop_time'];x=public[row['array_key']].astype(float);y=internal[pre+'_amp'].astype(float)
        assert x.shape==y.shape and np.isfinite(x).all() and np.all(x>=0)
        mask=np.zeros(len(y),dtype=bool);means_x=[];means_y=[];shapes=[]
        for n in internal[pre+'_notes']:
            s=max(0,int(float(n[0])/h));e=min(len(y),int(float(n[1])/h))
            if e>s:mask[s:e]=True
            if e-s>=4:
                means_x.append(x[s:e].mean());means_y.append(y[s:e].mean());q=corr(x[s:e],y[s:e])
                if q is not None:shapes.append(q)
        lx=np.log(x+1e-7);ly=np.log(y+1e-7)
        metrics={'raw':corr(x,y),'active_raw':corr(x[mask],y[mask]),'note_mean':corr(np.array(means_x),np.array(means_y)),
                 'within_note':float(np.mean(shapes)) if shapes else None,'log_corr':corr(lx,ly),
                 'log_smooth_1s':corr(gaussian_filter1d(lx,1/h,mode='reflect',truncate=4),gaussian_filter1d(ly,1/h,mode='reflect',truncate=4))}
        for k in keys:check(metrics[k],stored[t]['metrics'][k]);metric_checks+=1
        calculated[model][t]=metrics
    for label,subset in reports[model]['subsets'].items():
        keep=set(subset['tracks']);assert len(keep)==subset['n_tracks']
        for k in keys:
            values=[calculated[model][t][k] for t in keep if calculated[model][t][k] is not None]
            check(float(np.mean(values)),subset['summary'][model][k]['mean']);assert len(values)==subset['summary'][model][k]['n_tracks'];summary_checks+=1
        for pair,stats in subset['paired_group_bootstrap'].items():
            name=pair.split(' minus ')[1];ref={v['track']:v['views']['all'] for v in prior['per_track'][name]}
            for k in keys[:4]:
                ts=sorted(t for t in keep if calculated[model][t][k] is not None and ref[t][k] is not None)
                groups=sorted({tracks[t]['group'] for t in ts});diff=np.array([calculated[model][t][k]-ref[t][k] for t in ts])
                indices=[np.array([i for i,t in enumerate(ts) if tracks[t]['group']==g]) for g in groups]
                draws=np.random.RandomState(20260911).randint(len(groups),size=(5000,len(groups)))
                totals=np.array([diff[ix].sum() for ix in indices]);counts=np.array([len(ix) for ix in indices])
                boot=totals[draws].sum(1)/counts[draws].sum(1);expected=stats[k]
                check(diff.mean(),expected['track_mean_difference']);check(np.quantile(boot,[.025,.975]),expected['percentile_95_ci'])
                assert len(ts)==expected['paired_tracks'] and len(groups)==expected['groups'];pair_checks+=1
result={'status':'passed','public_tracks':{m:len(v) for m,v in calculated.items()},'per_track_metric_checks':metric_checks,
        'public_subset_summary_checks':summary_checks,'primary_paired_bootstrap_checks':pair_checks,'max_error':maximum,
        'scope':'Six public RMS views and primary paired intervals from archived RMS, against independently verified internal metrics; not synthesis or source-audio reproduction.'}
(P/'public_rms_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
