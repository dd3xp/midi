from pathlib import Path
import datetime,hashlib,json
import numpy as np
P=Path(__file__).resolve().parent;C=P.parent if (P.parent/'analysis_arrays.npz').exists() else P.parent/'revision_20260915/companion'
meta=json.loads((C/'tracks.json').read_text());a=np.load(C/'analysis_arrays.npz');old=json.loads((C/'level_and_aggregation_audit.json').read_text());base=json.loads((C/'expected_results.json').read_text());models=list(old['error_per_track'])
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'per_track':{},'summary':{},'comparisons':{},'sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [P/'LOG_ERROR_PLAN.md',P/'log_error_parts.py',C/'analysis_arrays.npz',C/'tracks.json',C/'level_and_aggregation_audit.json',C/'expected_results.json']}}
maxerr=0.;identityerr=0.
for m in models:
    vs=[]
    for row in meta:
        pre=row['prefix'];y=a[pre+'_amp'].astype(float);p=a[pre+'_'+m].astype(float);h=row['hop_time'];spans=[(max(0,int(float(n[0])/h)),min(len(y),int(float(n[1])/h))) for n in a[pre+'_notes']];spans=[(s,e) for s,e in spans if e-s>=4]
        errors=np.array([np.log(p[s:e].mean()+1e-7)-np.log(y[s:e].mean()+1e-7) for s,e in spans])
        b=float(errors.mean());c2=float(np.mean((errors-b)**2));E2=float(np.mean(errors**2));identityerr=max(identityerr,abs(E2-b*b-c2))
        # Independent second-moment identity and archived RMSE.
        assert abs(E2-b*b-c2)<1e-12
        er=abs(np.sqrt(E2)-old['error_per_track'][m][row['track']]['note_log_rmse']);maxerr=max(maxerr,er);assert er<1e-10
        vs.append({'track':row['track'],'b':b,'b2':b*b,'c2':c2,'c':float(np.sqrt(c2)),'E':float(np.sqrt(E2)),'E2':E2})
    out['per_track'][m]=vs
    out['summary'][m]={k:float(np.mean([v[k] for v in vs])) for k in ['b','b2','c2','c','E','E2']}
    out['summary'][m]['mean_abs_b']=float(np.mean([abs(v['b']) for v in vs]));out['summary'][m]['bias_squared_share']=sum(v['b2'] for v in vs)/sum(v['E2'] for v in vs)
for m in ['mamba','s4d','bigru']:
    for ref in ['instrument_ols','note_shape']:
        full=np.array([x['views']['all']['raw'] for x in base['per_track'][m]])-np.array([x['views']['all']['raw'] for x in base['per_track'][ref]])
        da={k:np.array([x[k] for x in out['per_track'][m]])-np.array([x[k] for x in out['per_track'][ref]]) for k in ['E','b2','c2','E2']}
        ix=(full>0)&(da['E']>0);bup=da['b2'][ix]>0;cup=da['c2'][ix]>0
        out['comparisons'][m+' minus '+ref]={'full_up_error_up':int(ix.sum()),'both_up':int((bup&cup).sum()),'bias_only_up':int((bup&~cup).sum()),'centered_only_up':int((~bup&cup).sum()),'neither_up':int((~bup&~cup).sum()),'mean_b2_change':float(da['b2'][ix].mean()),'mean_c2_change':float(da['c2'][ix].mean()),'mean_E2_change':float(da['E2'][ix].mean())}
out['verification']={'original_error_maximum_difference':maxerr,'decomposition_maximum_error':identityerr,'per_track_checks':len(meta)*len(models)}
(P/'log_error_parts.json').write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in out.items() if k in ['summary','comparisons','verification']},indent=2))
