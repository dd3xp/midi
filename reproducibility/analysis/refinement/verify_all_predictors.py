from pathlib import Path
import datetime,hashlib,json
import numpy as np
P=Path(__file__).resolve().parent;C=P.parent if (P.parent/'analysis_arrays.npz').exists() else P.parent/'revision_20260915/companion'
r=json.loads((P/'all_predictor_results.json').read_text());meta=json.loads((C/'tracks.json').read_text());a=np.load(C/'analysis_arrays.npz')
names=['mamba','s4d','bigru','instrument_ols','note_shape'];values={m:[] for m in names};maxerr=0.;checks=0
for row in meta:
    pre=row['prefix'];y=a[pre+'_amp'].astype(float);T=len(y);h=row['hop_time'];n=a[pre+'_notes']
    s=np.maximum(0,(n[:,0].astype(float)/h).astype(int));e=np.minimum(T,(n[:,1].astype(float)/h).astype(int));keep=e-s>=4;s=s[keep];e=e[keep];cs=np.r_[0.,y.cumsum()];ym=(cs[e]-cs[s])/(e-s)
    for m in names:
        p=a[pre+'_'+m].astype(float);cp=np.r_[0.,p.cumsum()];pm=(cp[e]-cp[s])/(e-s)
        values[m].append([float(np.corrcoef(p,y)[0,1]),float(np.corrcoef(pm,ym)[0,1]),float(np.sqrt(np.mean((np.log(pm+1e-7)-np.log(ym+1e-7))**2)))])
values={m:np.array(v) for m,v in values.items()}
def close(a,b):
    global maxerr,checks
    err=float(np.max(np.abs(np.asarray(a)-b)));maxerr=max(maxerr,err);checks+=1;assert err<1e-8,(a,b)
gs=sorted({x['group'] for x in meta});draw=np.random.RandomState(20260911).choice(len(gs),size=(5000,len(gs)),replace=True)
for pair,d in r['comparisons'].items():
    m,ref=pair.split(' minus ');v=values[m]-values[ref];up=v[:,0]>0;down=up&(v[:,1]<0);bad=up&(v[:,2]>0)
    close(up.sum(),d['full_improved']);close(down.sum(),d['note_decreased']);close(bad.sum(),d['error_increased'])
    close(np.quantile(-v[down,1],[.25,.5,.75]),d['note_loss_quartiles']);close(np.quantile(v[bad,2],[.25,.5,.75]),d['error_increase_quartiles'])
    close(np.quantile(v[down,0],[.25,.5,.75]),d['full_gain_quartiles_when_note_down'])
    counts=np.array([sum(up[i] for i,x in enumerate(meta) if x['group']==g) for g in gs])
    for name,mask in [('note',down),('error',bad)]:
        nums=np.array([sum(mask[i] for i,x in enumerate(meta) if x['group']==g) for g in gs]);den=counts[draw].sum(1)
        close(np.quantile(nums[draw].sum(1)/den,[.025,.975]),d[name+'_conditional']['ci95']);close(mask.sum()/up.sum(),d[name+'_conditional']['proportion'])
    for q in d['cutoffs']:
        c=q['cutoff'];close(((v[:,0]>c)&(v[:,1]<-c)).sum(),q['note_opposes']);close((v[:,0]>c).sum(),q['full_exceeds'])
for name,h in r['sha256'].items():
    p=P/name if (P/name).exists() else C/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==h,name
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'passed','numeric_checks':checks,'maximum_error':maxerr,
 'method':'Recompute correlations and note log errors from supplied arrays using prefix sums; resample group-level counts independently.',
 'results_sha256':hashlib.sha256((P/'all_predictor_results.json').read_bytes()).hexdigest()}
(P/'all_predictor_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
