"""Verify and export the completed fixed transfer test without new selection."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys
import numpy as np,torch
P=Path(__file__).resolve().parent;D=P/'musicnet'
r=json.loads((D/'results.json').read_text());meta=json.loads((D/'dataset_manifest.json').read_text())
rows=torch.load(D/'dataset.pt',weights_only=False);a=np.load(D/'predictions_all.npz')
assert len(rows)==21 and len({x['group'] for x in rows})==5
assert len({hashlib.sha256(x['amp'].tobytes()).hexdigest() for x in rows})==21
models=sorted(r['summary']);assert len(r['per_track'])==21*len(models)
assert len({(x['model'],x['track']) for x in r['per_track']})==len(r['per_track'])
lookup={(x['model'],x['track']):x for x in r['per_track']};max_error=0.
for row in rows:
    y=row['amp'].astype(float);hop=row['hop_time']
    spans=[(max(0,int(float(n[0])/hop)),min(len(y),int(float(n[1])/hop))) for n in row['notes']]
    spans=[(s,e) for s,e in spans if e-s>=4]
    ym=np.array([y[s:e].mean() for s,e in spans])
    for m in models:
        pred=a[row['track'].split('/')[-1]+'_'+m].astype(float)
        assert pred.shape==y.shape and np.isfinite(pred).all() and (pred>=0).all()
        pm=np.array([pred[s:e].mean() for s,e in spans]);d=np.log(pm+1e-7)-np.log(ym+1e-7)
        expected=dict(Error=float(np.linalg.norm(d)/np.sqrt(len(d))),b2=float(np.mean(d)**2),c2=float(np.var(d)))
        rr=lookup[m,row['track']]
        for k,v in expected.items():max_error=max(max_error,abs(v-rr[k]));assert abs(v-rr[k])<1e-10
        assert rr['eligible_notes']==len(spans)
for model,v in r['summary'].items():
    for k,stat in v.items():
        vals=[x[k] for x in r['per_track'] if x['model']==model and x[k] is not None]
        assert stat['n']==len(vals)
        if vals:assert abs(np.mean(vals)-stat['mean'])<1e-12
def fmt(x):
    return f'{x:.3f}'.replace('0.','.').replace('-0.','-.') if abs(x)<1 else f'{x:.3f}'
names={'instrument_ols':'Instrument OLS','note_ols':'Note OLS','note_shape':'Shared envelope','mamba':'Mamba','s4d':'S4D'}
lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Predictor & Full & Note & Error & $c^2$ \\',r'\midrule']
for m,name in names.items():lines.append(name+' & '+' & '.join(fmt(r['summary'][m][k]['mean']) for k in ['Full','Note','Error','c2'])+r' \\')
lines += [r'\bottomrule',r'\end{tabular}']
(D/'table_main.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
audit=dict(utc=datetime.now(timezone.utc).isoformat(),unique_recordings=21,composition_groups=5,
           prediction_models=len(models),model_recording_pairs=len(r['per_track']),
           all_waveform_horizons_preserved=True,finite_nonnegative_predictions=True,
           independent_error_recomputation_max_difference=max_error,
           separately_reported_from_original_186=True,seed=42,fold=0,
           dataset_sha256=meta['dataset_sha256'],results_sha256=hashlib.sha256((D/'results.json').read_bytes()).hexdigest(),
           group_sizes={g:sum(x['group']==g for x in rows) for g in r['groups']})
(D/'verification.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps(audit,indent=2))
