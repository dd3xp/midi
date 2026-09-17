"""Frozen references and diagnostics; validation selection precedes new predictions."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,sys
import numpy as np,torch
P=Path(__file__).resolve().parent;R=P.parents[1]/'local_run_20260912'
sys.path.insert(0,str(P.parent/'acceptance_cycle_20260916'))
from analyze_seeds import metrics
sys.path.insert(0,str(R/'code'))
from common import intervals,gate,metrics as scale_metrics
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def references(row,coef):
    ff=row['features'];counts=np.zeros(len(ff),np.float32);flat=counts.copy();shape=counts.copy()
    def exp(v):return np.exp(np.clip(v,-30,10)).astype(np.float32),int(np.sum((v<-30)|(v>10)))
    def frame(c):return exp(ff@np.array(c['coef'])+c['intercept'])
    glob,gclip=frame(coef['global']);inst,iclip=frame(coef['instrument'].get(row['instrument'],coef['global']))
    spans=list(intervals(row,1));xx=np.array([ff[s:e].mean(axis=0) for s,e in spans],dtype=float)
    levels,nclip=exp(xx@np.array(coef['note']['coef'])+coef['note']['intercept'])
    template=np.array(coef['shape_templates'].get(row['instrument'],coef['shape_templates']['global']))
    for (s,e),level in zip(spans,levels):
        curve=np.interp(np.linspace(0,1,e-s),np.linspace(0,1,64),template)
        curve/=max(float(curve.mean()),1e-7)
        flat[s:e]+=level;shape[s:e]+=level*curve;counts[s:e]+=1
    return dict(global_ols=glob,instrument_ols=inst,note_ols=flat/np.maximum(counts,1),
                note_shape=shape/np.maximum(counts,1),gate=gate(row).astype(np.float32)),dict(global_ols=gclip,instrument_ols=iclip,note_ols=nclip)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--role',choices=['validation','musicnet'],required=True);args=ap.parse_args()
    frozen=json.loads((P/'confirmation_frozen.json').read_text())
    cp=R/'baselines/fold0/coefficients.json';assert sha(cp)==frozen['original_inputs']['baselines/fold0/coefficients.json']
    coef=json.loads(cp.read_text());fold=json.loads((R/'artifacts/folds.json').read_text())[0]
    original=torch.load(R/'artifacts/dataset_corrected.pt',weights_only=False)
    assert set(coef['train_tracks'])=={original[i]['track'] for i in fold['train']}
    dest=P/'musicnet';dest.mkdir(exist_ok=True)
    if args.role=='validation':
        assert not (dest/'selection.json').exists(),'Do not overwrite frozen selection'
        rows=[original[i] for i in fold['validation']]
        dirs={m:P/f'validation_{m}_verified' for m in ['mamba','s4d']}
    else:
        assert (dest/'selection.json').exists()
        rows=torch.load(dest/'dataset.pt',weights_only=False);dirs={m:P/f'musicnet_{m}' for m in ['mamba','s4d']}
    predictions={};input_hashes={}
    for m,d in dirs.items():
        st=json.loads((d/'status.json').read_text());assert st['state']=='complete'
        assert sha(d/'predictions.pt')==st['prediction_sha256']
        pred=torch.load(d/'predictions.pt',weights_only=False)
        assert len(pred)==len(rows) and {p['track'] for p in pred}=={r['track'] for r in rows}
        predictions[m]={p['track']:p['amp_pred'] for p in pred};input_hashes[m]=sha(d/'predictions.pt')
    records=[];arrays={}
    for row in rows:
        refs,clipped=references(row,coef)
        for m in predictions:refs[m]=predictions[m][row['track']]
        for m,p in refs.items():
            v=metrics(p.astype(float),row['amp'].astype(float),row['notes'],row['hop_time'])
            records.append(dict(track=row['track'],group=row['group'],model=m,clipped=clipped.get(m,0),**v))
            if args.role=='musicnet':
                views=scale_metrics(p.astype(float),row,sensitivities=False)
                records[-1]['fixed_output_scales']={k:value for k,value in views.items() if k.startswith(('log_corr','log_smooth_','linear_smooth_'))}
            arrays[row['track'].split('/')[-1]+'_'+m]=p
    candidates=['instrument_ols','mamba','note_ols','note_shape','s4d'];measures=['Full','Active','Note','Within','Error','b2','c2']
    summaries={m:{k:dict(mean=float(np.mean(v)) if v else None,n=len(v)) for k in measures for v in [[r[k] for r in records if r['model']==m and r[k] is not None]]} for m in sorted({r['model'] for r in records})}
    result=dict(utc=datetime.now(timezone.utc).isoformat(),role=args.role,input_hashes=input_hashes,per_track=records,summary=summaries)
    if args.role=='validation':
        eligible=[m for m in candidates if all(summaries[m][k]['n']==len(rows) for k in ['Full','Note'])]
        assert eligible
        selected={}
        for k in ['Full','Note']:
            best=max(summaries[m][k]['mean'] for m in eligible)
            selected[k]=min(m for m in eligible if best-summaries[m][k]['mean']<=1e-12)
        result.update(selected=selected,candidates=candidates,eligible=eligible,validation_tracks=[r['track'] for r in rows],
                      plan_sha256=frozen['plan_sha256'],new_test_outputs_seen=False)
        (dest/'selection.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps(dict(selected=selected,summary=summaries),indent=2))
    else:
        groups=sorted({r['group'] for r in rows});rng=np.random.default_rng(20260916)
        draws=rng.integers(0,len(groups),size=(5000,len(groups)))
        lookup={(r['model'],r['track']):r for r in records};pairs={}
        for model,ref in [('mamba','instrument_ols'),('mamba','note_shape'),('s4d','instrument_ols'),('s4d','note_shape'),('note_shape','note_ols'),('s4d','mamba')]:
            key=model+' minus '+ref;deltas={};bywork={}
            for k in measures:
                workvals=[[lookup[model,r['track']][k]-lookup[ref,r['track']][k] for r in rows if r['group']==g and lookup[model,r['track']][k] is not None and lookup[ref,r['track']][k] is not None] for g in groups]
                flat=[v for vs in workvals for v in vs]
                bootstrap=[float(np.mean([v for ix in draw for v in workvals[ix]])) for draw in draws if any(workvals[ix] for ix in draw)]
                deltas[k]=dict(mean=float(np.mean(flat)) if flat else None,n=len(flat),
                     exploratory_group_ci=np.percentile(bootstrap,[2.5,97.5]).tolist() if bootstrap else None,
                     equal_work_mean=float(np.mean([np.mean(vs) for vs in workvals if vs])) if flat else None)
                bywork[k]={g:float(np.mean(vs)) if vs else None for g,vs in zip(groups,workvals)}
            up=[r for r in rows if lookup[model,r['track']]['Full']>lookup[ref,r['track']]['Full']]
            discord=[r['track'] for r in up if lookup[model,r['track']]['Note']<lookup[ref,r['track']]['Note']]
            pairs[key]=dict(deltas=deltas,by_work=bywork,full_improved=len(up),note_decreased=len(discord),discordant_tracks=discord)
        work_summary={m:{g:{k:float(np.mean(v)) if v else None for k in measures for v in [[r[k] for r in records if r['model']==m and r['group']==g and r[k] is not None]]} for g in groups} for m in summaries}
        equal_work_summary={m:{k:float(np.mean(v)) if v else None for k in measures for v in [[work_summary[m][g][k] for g in groups if work_summary[m][g][k] is not None]]} for m in summaries}
        result.update(groups=groups,pairs=pairs,selection=json.loads((dest/'selection.json').read_text())['selected'],
                      work_summary=work_summary,equal_work_summary=equal_work_summary,
                      frozen_plan_sha256=frozen['plan_sha256'],analysis_sha256=sha(Path(__file__)),
                      dataset_sha256=sha(dest/'dataset.pt'),selection_sha256=sha(dest/'selection.json'))
        (dest/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        np.savez_compressed(dest/'predictions_all.npz',**arrays)
        print(json.dumps(dict(summary=summaries,pairs=pairs),indent=2))

if __name__=='__main__':main()
