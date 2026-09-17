"""Full-coverage per-seed diagnostics; no fitting or seed selection."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json
import numpy as np
import torch
torch.set_num_threads(2)
P=Path(__file__).resolve().parent;ROOT=P.parents[1];C=P.parent/'revision_20260915/companion'
if (P.parent/'analysis_arrays.npz').exists():C=P.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def corr(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if len(a)<3 or a.std()<1e-10 or b.std()<1e-10:return None
    a=a-a.mean();b=b-b.mean();return float(a@b/np.sqrt((a@a)*(b@b)))

def metrics(p,y,notes,hop):
    assert p.shape==y.shape and np.isfinite(p).all() and (p>=0).all()
    active=np.zeros(len(y),bool);means_p=[];means_y=[];shapes=[]
    for n in notes:
        s=max(0,int(float(n[0])/hop));e=min(len(y),int(float(n[1])/hop))
        if e>s:active[s:e]=True
        if e-s<4:continue
        means_p.append(p[s:e].mean());means_y.append(y[s:e].mean())
        v=corr(p[s:e],y[s:e])
        if v is not None:shapes.append(v)
    d=np.log(np.array(means_p)+1e-7)-np.log(np.array(means_y)+1e-7)
    b=d.mean();c2=((d-b)**2).mean();e2=(d*d).mean()
    assert abs(e2-b*b-c2)<1e-10
    return dict(Full=corr(p,y),Active=corr(p[active],y[active]),Note=corr(means_p,means_y),
                Within=float(np.mean(shapes)) if shapes else None,Error=float(np.sqrt(e2)),
                b2=float(b*b),c2=float(c2),eligible_notes=len(d),defined_shape_notes=len(shapes))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['mamba','s4d'],required=True)
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--seeds',nargs='+',type=int,default=[123,456])
    ap.add_argument('--execution',type=Path,help='Extracted cached-input execution companion')
    ap.add_argument('--output',type=Path);args=ap.parse_args()
    ts=json.loads((C/'tracks.json').read_text());a=np.load(C/'analysis_arrays.npz');n=len(ts);assert n==186
    source=json.loads((P/'seed_source_hashes.json').read_text());old=ROOT/'local_run_20260912'
    cache_dir=args.execution/'data' if args.execution else old/'artifacts'
    original_protocol=args.execution/'mamba_s4d/protocol.json' if args.execution else old/'artifacts/protocol.json'
    assert sha(cache_dir/'dataset_corrected.pt')==source['artifacts/dataset_corrected.pt']
    assert sha(original_protocol)==source['artifacts/protocol.json']
    rows=torch.load(cache_dir/'dataset_corrected.pt',map_location='cpu',weights_only=False)
    folds=json.loads((cache_dir/'folds.json').read_text());assert folds==json.loads((C/'folds.json').read_text())
    for i,t in enumerate(ts):
        assert rows[i]['track']==t['track'] and rows[i]['group']==t['group']
        for key in ['amp','notes','features']:assert np.array_equal(rows[i][key],a[t['prefix']+'_'+key])
        assert rows[i]['hop_time']==t['hop_time']
    seen=[]
    for f in folds:
        roles=[set(f[k]) for k in ['train','validation','evaluation']]
        assert set.union(*roles)==set(range(n)) and sum(map(len,roles))==n
        for i in range(3):
            for j in range(i+1,3):assert not {ts[k]['group'] for k in roles[i]}&{ts[k]['group'] for k in roles[j]}
        seen.extend(f['evaluation'])
    assert sorted(seen)==list(range(n))
    fixed={};oldstats=json.loads((C/'expected_results.json').read_text())['per_track']
    olderr=json.loads((P/'corpus_results.json').read_text())['per_track'];maxerr=0.
    for m in [args.model,'instrument_ols','note_shape']:
        fixed[m]=[];lookup={r['track']:r['views']['all'] for r in oldstats[m]}
        errors={r['track']:r for r in olderr[m]}
        for t in ts:
            pre=t['prefix'];v=metrics(a[pre+'_'+m].astype(float),a[pre+'_amp'].astype(float),a[pre+'_notes'],t['hop_time'])
            for new,oldkey in [('Full','raw'),('Active','active_raw'),('Note','note_mean'),('Within','within_note')]:
                x=lookup[t['track']][oldkey];assert (x is None)==(v[new] is None)
                if x is not None:maxerr=max(maxerr,abs(x-v[new]));assert abs(x-v[new])<1e-8
            for key in ['Error','b2','c2']:assert abs(v[key]-errors[t['track']][key])<1e-8
            fixed[m].append(dict(track=t['track'],**v))
    frozen=json.loads((args.root/'frozen_hashes.json').read_text());plan=json.loads((args.root/'plan.json').read_text())
    assert plan['original_source_hashes']==source
    for rel,h in frozen.items():
        if (args.root/rel).exists():assert sha(args.root/rel)==h,rel
        else:assert rel.endswith('dataset_corrected.pt') and h==source['artifacts/dataset_corrected.pt'],rel
    if (args.root/'export_manifest.json').exists():
        exp=json.loads((args.root/'export_manifest.json').read_text())
        for rel,h in exp['files'].items():assert sha(args.root/rel)==h,rel
        assert exp['frozen_files_verified']==len(frozen)
    runs={42:fixed[args.model]};integrity={};input_hashes={}
    for seed in args.seeds:
        run=args.root/f'seed{seed}';protocol=json.loads((run/'artifacts/protocol.json').read_text())
        assert protocol['model_seed']==seed
        baseline=json.loads(original_protocol.read_text())
        for k,v in baseline.items():
            if k not in ['model_seed','version','created_utc','gpu_sharing']:assert protocol[k]==v,k
        assert protocol['parent_protocol_sha256']==source['artifacts/protocol.json']
        predictions={};jobs=[];saved_metrics={}
        for fold,f in enumerate(folds):
            job=run/'jobs'/f'{args.model}_fold{fold}'
            done=json.loads((job/'complete.json').read_text());cfg=json.loads((job/'config.json').read_text())
            assert done['status']=='complete' and cfg['seed']==seed and cfg['allocator_fraction_limit']<=.8
            assert cfg['split_counts']=={k:len(f[k]) for k in ['train','validation','evaluation']}
            assert cfg['protocol_sha256']==hashlib.sha256((run/'artifacts/protocol.json').read_bytes()+(run/'artifacts/folds.json').read_bytes()).hexdigest()
            assert cfg['code_sha256']==source['code/train.py']
            history=json.loads((job/'history.json').read_text());assert len(history)==done['completed_epochs']
            assert [x['epoch'] for x in history]==list(range(1,len(history)+1))
            assert all(np.isfinite([r['train_loss'],r['validation_loss'],r['seconds']]).all() for r in history)
            assert done['best_epoch']==min(history,key=lambda x:x['validation_loss'])['epoch']
            weights=torch.load(job/'best.pt',map_location='cpu',weights_only=True)
            assert all(torch.isfinite(v).all().item() for v in weights.values())
            saved=torch.load(job/'predictions.pt',map_location='cpu',weights_only=False)
            assert len(saved)==done['n_evaluation_tracks']==len(f['evaluation'])
            assert {r['track'] for r in saved}=={ts[i]['track'] for i in f['evaluation']}
            for r in saved:
                assert r['track'] not in predictions;predictions[r['track']]=r['amp_pred']
            for row in json.loads((job/'metrics.json').read_text())['per_track']:
                assert row['track'] not in saved_metrics;saved_metrics[row['track']]=row['metrics']
            input_hashes[str(job/'predictions.pt')]=sha(job/'predictions.pt')
            jobs.append(dict(fold=fold,tracks=len(saved),epochs=len(history),best_epoch=done['best_epoch'],
                median_epoch_seconds=float(np.median([r['seconds'] for r in history])),
                peak_gpu_gib=max(r['peak_gpu_gib'] for r in history),checkpoint_sha256=sha(job/'best.pt')))
        assert len(predictions)==n
        vals=[];new_maxerr=0.
        for t in ts:
            pre=t['prefix'];v=metrics(np.asarray(predictions[t['track']],dtype=float),a[pre+'_amp'].astype(float),a[pre+'_notes'],t['hop_time'])
            for key,oldkey in [('Full','raw'),('Active','active_raw'),('Note','note_mean'),('Within','within_note')]:
                x=saved_metrics[t['track']][oldkey];assert (x is None)==(v[key] is None)
                if x is not None:new_maxerr=max(new_maxerr,abs(x-v[key]));assert abs(x-v[key])<1e-8
            vals.append(dict(track=t['track'],**v))
        runs[seed]=vals;integrity[seed]=dict(tracks=n,unique_tracks=n,folds_disjoint=True,predictions_finite_nonnegative=True,stored_metrics_max_error=new_maxerr,jobs=jobs)
        print('Validated',args.model,seed,flush=True)
    groupnames=sorted({t['group'] for t in ts});assert len(groupnames)==44
    draw=np.random.RandomState(20260911).randint(44,size=(5000,44))
    wg=np.stack([(draw==g).sum(1) for g in range(44)],axis=1);w=wg[:,[groupnames.index(t['group']) for t in ts]]
    result={}
    for seed,rs in runs.items():
        summary={k:dict(mean=float(np.mean([r[k] for r in rs if r[k] is not None])),tracks=sum(r[k] is not None for r in rs)) for k in ['Full','Active','Note','Within','Error']}
        pairs={}
        for ref in ['instrument_ols','note_shape']:
            ds={k:np.array([r[k]-q[k] if r[k] is not None and q[k] is not None else np.nan for r,q in zip(rs,fixed[ref])]) for k in ['Full','Active','Note','Within','Error','b2','c2']}
            paired={}
            for k,d in ds.items():
                mask=np.isfinite(d);weights=w[:,mask]
                available_groups=sorted({ts[i]['group'] for i in np.flatnonzero(mask)})
                if len(available_groups)!=44:
                    rd=np.random.RandomState(20260911).randint(len(available_groups),size=(5000,len(available_groups)))
                    gw=np.stack([(rd==g).sum(1) for g in range(len(available_groups))],axis=1)
                    weights=gw[:,[available_groups.index(ts[i]['group']) for i in np.flatnonzero(mask)]]
                valid=weights.sum(1)>0
                boot=weights[valid]@d[mask]/weights[valid].sum(1)
                paired[k]=dict(mean=float(d[mask].mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),tracks=int(mask.sum()),groups=len(available_groups))
            up=ds['Full']>0;nd=up&(ds['Note']<0);eu=up&(ds['Error']>0)
            comparisons=dict(full_up=int(up.sum()),note_down=int(nd.sum()),error_up=int(eu.sum()),
                note_loss_median=float(np.median(-ds['Note'][nd])) if nd.any() else None,
                full_gain_median_on_note_down=float(np.median(ds['Full'][nd])) if nd.any() else None,
                error_rise_median=float(np.median(ds['Error'][eu])) if eu.any() else None)
            for label,mask in [('note_down',nd),('error_up',eu)]:
                den=w@up;valid=den>0
                comparisons[label+'_conditional_ci95']=np.quantile((w@mask)[valid]/den[valid],[.025,.975]).tolist() if valid.any() else None
            pairs[ref]=dict(**comparisons,paired=paired)
        result[seed]=dict(summary=summary,comparisons=pairs,per_track=rs)
    out=dict(utc=datetime.now(timezone.utc).isoformat(),model=args.model,seeds=list(runs),tracks_per_seed=n,groups=44,
        uncertainty='Group bootstrap conditional on each seed and fixed folds; no pooling of seed-track copies.',
        seed42_recomputation_max_error=maxerr,integrity=integrity,results=result,
        sha256=dict(input_hashes,analysis_arrays=sha(C/'analysis_arrays.npz'),script=sha(Path(__file__)),plan=sha(P/'PLAN.md')))
    dest=args.output or P/f'{args.model}_seed_results.json';dest.write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({s:dict(summary=r['summary'],directions={ref:{k:v for k,v in x.items() if k!='paired'} for ref,x in r['comparisons'].items()}) for s,r in result.items()},indent=2))

if __name__=='__main__':main()
