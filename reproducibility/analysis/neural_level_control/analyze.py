from pathlib import Path
import datetime,hashlib,json,time
import numpy as np

P=Path(__file__).resolve().parent;C=P.parent/'revision_20260915/companion'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
freeze=json.loads((P/'input_manifest.json').read_text())
assert sha(P/'PLAN.md')==freeze['plan_sha256']
assert all(sha(C/k)==v for k,v in freeze['input_sha256'].items())
assert not (P/'results.json').exists(), 'Preserve previous results; do not overwrite'
started=time.perf_counter()
launch=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command='python paper1/strong_level_control_20260918/analyze.py',plan_sha256=sha(P/'PLAN.md'),code_sha256=sha(Path(__file__)),input_manifest_sha256=sha(P/'input_manifest.json'))
(P/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')

def corr(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)<3 or x.std()<1e-10 or y.std()<1e-10:return None
    value=float(np.corrcoef(x,y)[0,1])
    xc=x-x.mean();yc=y-y.mean()
    independent=float(np.dot(xc,yc)/np.sqrt(np.dot(xc,xc)*np.dot(yc,yc)))
    assert abs(value-independent)<1e-12
    return value

def metrics(pred,target,eligible):
    pm=np.array([pred[s:e].mean() for s,e in eligible]);tm=np.array([target[s:e].mean() for s,e in eligible])
    within=[corr(pred[s:e],target[s:e]) for s,e in eligible]
    valid=[v for v in within if v is not None]
    return dict(Full=corr(pred,target),Note=corr(pm,tm),Error=float(np.sqrt(np.mean((np.log(pm+1e-7)-np.log(tm+1e-7))**2))),Within=float(np.mean(valid)) if valid else None,Within_defined_notes=len(valid)),pm

a=np.load(C/'analysis_arrays.npz');rows=[];saved={};old_full=[]
max_mean_relative=0.;max_note_difference=0.;max_error_difference=0.;max_shape_crosscheck=0.
for t in freeze['selected_tracks']:
    pre=t['prefix'];target=a[pre+'_amp'].astype(float);notes=t['spans'];eligible=[(s,e) for s,e in notes if e-s>=4]
    assert len(target)==t['frames'] and len(eligible)>=3
    note_ols=a[pre+'_note_ols'].astype(float);shared=a[pre+'_note_shape'].astype(float)
    old_full.append([corr(note_ols,target),corr(shared,target)])
    templates=[]
    for s,e in notes:
        base=note_ols[s:e];shape=shared[s:e]
        assert np.isfinite(shape).all() and (shape>=0).all() and shape.mean()>0
        assert (base>0).all() and np.all(base==base[0])
        q=shape/shape.mean()
        independent=shape/base;independent/=independent.mean()
        max_shape_crosscheck=max(max_shape_crosscheck,float(np.max(np.abs(q-independent))))
        assert np.allclose(q,independent,rtol=1e-12,atol=1e-12)
        templates.append(q)
    for model in ['mamba','s4d']:
        original=a[pre+'_'+model].astype(float)
        assert len(original)==len(target) and np.isfinite(original).all() and (original>=0).all()
        flat=original.copy();shaped=original.copy();covered=np.zeros(len(target),bool)
        for (s,e),q in zip(notes,templates):
            mean=float(original[s:e].mean());flat[s:e]=mean;shaped[s:e]=mean*q;covered[s:e]=True
            # The identity shape leaves exactly the same constant reference.
            assert np.array_equal(flat[s:e],mean*np.ones(e-s))
            for rendered in [flat,shaped]:
                deviation=abs(float(rendered[s:e].mean())-mean)/max(mean,1e-7)
                max_mean_relative=max(max_mean_relative,deviation)
                assert deviation<=1e-10
        assert np.array_equal(original[~covered],flat[~covered]) and np.array_equal(original[~covered],shaped[~covered])
        assert np.isfinite(flat).all() and np.isfinite(shaped).all()
        mm={};pms={}
        for name,pred in [('original',original),('flat',flat),('shared',shaped)]:
            mm[name],pms[name]=metrics(pred,target,eligible)
            assert all(mm[name][k] is not None for k in ['Full','Note','Error'])
        assert mm['flat']['Within'] is None
        for name in ['flat','shared']:
            nd=abs(mm[name]['Note']-mm['original']['Note']);ed=abs(mm[name]['Error']-mm['original']['Error'])
            assert nd<=1e-10 and ed<=1e-10
            max_note_difference=max(max_note_difference,nd);max_error_difference=max(max_error_difference,ed)
        rows.append(dict(track=t['track'],prefix=pre,group=t['group'],dataset=t['dataset'],model=model,frames=len(target),eligible_notes=len(eligible),metrics=mm,Full_difference=mm['shared']['Full']-mm['flat']['Full']))
        saved[pre+'_'+model+'_flat']=flat;saved[pre+'_'+model+'_shared']=shaped

old_full=np.mean(old_full,axis=0);assert [round(float(v),3) for v in old_full]==[.467,.546]
groups=sorted({t['group'] for t in freeze['selected_tracks']});assert len(groups)==34
draws=np.random.default_rng(20260918).integers(0,len(groups),(5000,len(groups)))
summaries={}
for model in ['mamba','s4d']:
    subset=[r for r in rows if r['model']==model];assert len(subset)==len({r['track'] for r in subset})==133
    values=np.array([r['Full_difference'] for r in subset]);group_values=[[r['Full_difference'] for r in subset if r['group']==g] for g in groups]
    sums=np.array([sum(v) for v in group_values]);counts=np.array([len(v) for v in group_values])
    boot=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
    agg={}
    for condition in ['original','flat','shared']:
        agg[condition]={}
        for metric in ['Full','Note','Within','Error']:
            valid=[r['metrics'][condition][metric] for r in subset if r['metrics'][condition][metric] is not None]
            agg[condition][metric]=dict(mean=float(np.mean(valid)) if valid else None,tracks=len(valid))
    summaries[model]=dict(tracks=133,groups=34,eligible_notes=sum(r['eligible_notes'] for r in subset),means=agg,Full_shared_minus_flat=dict(mean=float(values.mean()),paired_group_interval=np.percentile(boot,[2.5,97.5]).tolist(),positive_tracks=int((values>0).sum()),negative_tracks=int((values<0).sum()),zero_tracks=int((values==0).sum())))
np.savez_compressed(P/'controlled_predictions.npz',**saved)
out=dict(**launch,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started,scope='Prespecified additional retrospective controls on seed42 predictions; not new independent validation',input_sha256=freeze['input_sha256'],selected_tracks=133,groups=34,bootstrap=dict(draws=5000,seed=20260918,unit='whole group',weight='equal track'),original_Note_OLS_Full=float(old_full[0]),original_shared_envelope_Full=float(old_full[1]),max_mean_relative_error=max_mean_relative,max_Note_difference=max_note_difference,max_Error_difference=max_error_difference,max_template_recovery_difference=max_shape_crosscheck,controlled_arrays_sha256=sha(P/'controlled_predictions.npz'),summary=summaries,per_track=rows)
(P/'results.json').write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf-8')
manifest={f.name:sha(f) for f in sorted(P.iterdir()) if f.is_file() and f.name!='manifest.json'}
(P/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k not in ['per_track','input_sha256']},indent=2))
