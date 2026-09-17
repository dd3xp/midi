from pathlib import Path
import numpy as np,json,hashlib,datetime
P=Path(__file__).resolve().parent;C=P.parent/'revision_20260915/companion'
if (P.parent/'analysis_arrays.npz').exists():C=P.parent
ts=json.loads((C/'tracks.json').read_text());a=np.load(C/'analysis_arrays.npz')
def corr(x,y):
    if len(x)<3 or np.std(x)<1e-10 or np.std(y)<1e-10:return None
    return float(np.corrcoef(x,y)[0,1])
def quant(x):return dict(zip(['median','q95','max'],np.quantile(x,[.5,.95,1]).tolist()))
records=[];al=[];non=[];all_log=[];non_log=[];direct_error=0.
for t in ts:
    pre=t['prefix'];y=a[pre+'_amp'].astype(float);no=a[pre+'_note_ols'].astype(float);sh=a[pre+'_note_shape'].astype(float)
    notes=a[pre+'_notes'];hop=t['hop_time'];counts=np.zeros(len(y),int);spans=[]
    for n in notes:
        s=max(0,int(float(n[0])/hop));e=min(len(y),int(float(n[1])/hop));spans.append((s,e))
        if e>s:counts[s:e]+=1
    eligible=[(s,e) for s,e in spans if e-s>=4]
    keep=np.array([bool((counts[s:e]<=1).all()) for s,e in eligible])
    means=[]
    for v in [y,no,sh]:
        direct=np.array([v[s:e].mean() for s,e in eligible]);cs=np.r_[0.,np.cumsum(v)]
        sums=np.array([(cs[e]-cs[s])/(e-s) for s,e in eligible]);direct_error=max(direct_error,float(np.max(abs(direct-sums))))
        assert np.allclose(direct,sums,rtol=1e-8,atol=1e-10)
        means.append(direct)
    ym,nm,sm=means;relative=abs(sm-nm)/np.maximum(nm,1e-7);dl=abs(np.log(sm+1e-7)-np.log(nm+1e-7))
    al.extend(relative);non.extend(relative[keep]);all_log.extend(dl);non_log.extend(dl[keep])
    error=lambda x:float(np.sqrt(np.mean((np.log(x+1e-7)-np.log(ym+1e-7))**2)))
    record=dict(track=t['track'],eligible=len(eligible),nonoverlap_notes=int(keep.sum()),no_overlapping_frames=bool(counts.max()<=1),
       relative_mean_change=quant(relative),Note_ols=corr(nm,ym),Note_shape=corr(sm,ym),
       Full_ols=corr(no,y),Full_shape=corr(sh,y),Error_ols=error(nm),Error_shape=error(sm))
    records.append(record)
subset=[x for x in records if x['no_overlapping_frames']]
out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=hashlib.sha256((P/'TEMPLATE_AUDIT_PLAN.md').read_bytes()).hexdigest(),
    arrays_sha256=hashlib.sha256((C/'analysis_arrays.npz').read_bytes()).hexdigest(),tracks=186,notes=len(al),nonoverlap_notes=len(non),
    all_relative=quant(al),nonoverlap_relative=quant(non),all_log=quant(all_log),nonoverlap_log=quant(non_log),
    independent_means_max_error=direct_error,per_track=records,
    absolute_track_Note_change=quant([abs(x['Note_shape']-x['Note_ols']) for x in records]),
    absolute_track_Error_change=quant([abs(x['Error_shape']-x['Error_ols']) for x in records]),
    fully_nonoverlapping_tracks=dict(n=len(subset),means={k:float(np.mean([x[k] for x in subset])) for k in ['Full_ols','Full_shape','Note_ols','Note_shape','Error_ols','Error_shape']}))
expected=P/'template_preservation.json'
if expected.exists():
    old=json.loads(expected.read_text())
    assert {k:v for k,v in out.items() if k!='utc'}=={k:v for k,v in old.items() if k!='utc'}
    target=P/'template_preservation_recomputed.json'
else:target=expected
target.write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps({k:v for k,v in out.items() if k!='per_track'},indent=2))
