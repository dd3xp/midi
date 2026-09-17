"""Matched OLS/event/shape references, plus historical OLS reconstruction."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
import sys
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from common import RUN, ART, save_json, metrics, summarize, gate, intervals

sys.path.insert(0,str(RUN/'source_snapshot'))
from src.model.solo.dataset import notes_to_frame_features

rows=torch.load(ART/'dataset_corrected.pt',map_location='cpu',weights_only=False)
folds=json.loads((ART/'folds.json').read_text())
historical={'train':[i for i,r in enumerate(rows) if r['historical_split']=='train'],
            'evaluation':[i for i,r in enumerate(rows) if r['historical_split']=='evaluation']}
affected={r['track'] for r in json.loads((ART/'input_audit_all_tracks.json').read_text())['affected_tracks']}

def design(row,legacy=False):
    if legacy and row['track'] in affected:
        return notes_to_frame_features(row['notes'],len(row['amp']),row['hop_time'],n_features=20,instrument=row['instrument'])
    return row['features']

def fit_frame(ids,legacy=False):
    X=np.concatenate([design(rows[i],legacy) for i in ids])
    y=np.concatenate([np.log(np.clip(rows[i]['amp'],1e-7,None)).astype(np.float32) for i in ids])
    return LinearRegression().fit(X,y)

def safe_exp(x):
    # Guard catastrophic extrapolation explicitly and report clipping counts.
    count=int(np.sum((x<-30)|(x>10)))
    return np.exp(np.clip(x,-30,10)).astype(np.float32),count

def fit_event(ids):
    X,y,shapes,inst_shapes=[],[],[],{}
    phase=np.linspace(0,1,64)
    for i in ids:
        r=rows[i]
        for start,end in intervals(r):
            a=r['amp'][start:end];mean=float(a.mean())
            X.append(r['features'][start:end].mean(axis=0));y.append(np.log(max(mean,1e-7)))
            if mean>1e-7:
                shape=np.interp(phase,np.linspace(0,1,len(a)),a/mean)
                shapes.append(shape);inst_shapes.setdefault(r['instrument'],[]).append(shape)
    reg=LinearRegression().fit(np.asarray(X,dtype=np.float64),np.asarray(y,dtype=np.float64))
    shape_lookup={k:np.mean(v,axis=0) for k,v in inst_shapes.items()}
    shape_lookup['global']=np.mean(shapes,axis=0)
    return reg,shape_lookup

def event_predict(row,reg,templates):
    flat=np.zeros(len(row['amp']),dtype=np.float32);shaped=flat.copy();counts=flat.copy()
    spans=list(intervals(row,1))
    X=np.asarray([row['features'][s:e].mean(axis=0) for s,e in spans],dtype=float)
    levels,clipped=safe_exp(reg.predict(X))
    template=templates.get(row['instrument'],templates['global'])
    for (start,end),level in zip(spans,levels):
        shape=np.interp(np.linspace(0,1,end-start),np.linspace(0,1,64),template)
        shape=shape/max(float(shape.mean()),1e-7)
        flat[start:end]+=level;shaped[start:end]+=level*shape;counts[start:end]+=1
    flat/=np.maximum(counts,1);shaped/=np.maximum(counts,1)
    return flat,shaped,clipped

partitions=[('historical_original_features',historical,True),('historical_corrected_features',historical,False)]
partitions += [(f'fold{f}',split,False) for f,split in enumerate(folds)]
for name,split,legacy in partitions:
    destination=RUN/'baselines'/name;destination.mkdir(parents=True,exist_ok=True)
    if (destination/'complete.json').exists():continue
    print('Fitting '+name,flush=True)
    global_reg=fit_frame(split['train'],legacy)
    instruments=sorted({rows[i]['instrument'] for i in split['train']})
    local={inst:fit_frame([i for i in split['train'] if rows[i]['instrument']==inst],legacy) for inst in instruments}
    event_reg,templates=fit_event(split['train']) if not legacy else (None,None)
    model_names=['gate','global_ols','instrument_ols']+([] if legacy else ['note_ols','note_shape'])
    scores={key:[] for key in model_names};preds={key:[] for key in model_names}
    clipping={key:0 for key in model_names};fallback=[]
    for i in split['evaluation']:
        row=rows[i];X=design(row,legacy)
        global_pred,cg=safe_exp(global_reg.predict(X))
        per_inst=local.get(row['instrument'],global_reg)
        if row['instrument'] not in local:fallback.append(row['track'])
        local_pred,cl=safe_exp(per_inst.predict(X))
        candidates={'gate':gate(row).astype(np.float32),'global_ols':global_pred,'instrument_ols':local_pred}
        clipping['global_ols']+=cg;clipping['instrument_ols']+=cl
        if not legacy:
            flat,shape,ce=event_predict(row,event_reg,templates)
            candidates.update(note_ols=flat,note_shape=shape)
            clipping['note_ols']+=ce;clipping['note_shape']+=ce
        for model,pred in candidates.items():
            scores[model].append({'track':row['track'],'group':row['group'],'instrument':row['instrument'],'metrics':metrics(pred,row)})
            preds[model].append({'track':row['track'],'amp_pred':pred})
    for model in model_names:
        torch.save(preds[model],destination/f'{model}_predictions.pt')
        save_json(destination/f'{model}_metrics.json',{'summary':summarize(scores[model]),'per_track':scores[model],
                  'clipped_log_predictions':clipping[model],'instrument_fallback_tracks':fallback if model=='instrument_ols' else []})
    coefficients={'global':{'coef':global_reg.coef_.tolist(),'intercept':float(global_reg.intercept_)},
                  'instrument':{key:{'coef':reg.coef_.tolist(),'intercept':float(reg.intercept_)} for key,reg in local.items()},
                  'train_tracks':[rows[i]['track'] for i in split['train']]}
    if not legacy:
        coefficients['note']={'coef':event_reg.coef_.tolist(),'intercept':float(event_reg.intercept_)}
        coefficients['shape_templates']={key:value.tolist() for key,value in templates.items()}
    save_json(destination/'coefficients.json',coefficients)
    save_json(destination/'complete.json',{'status':'complete','n_train':len(split['train']),'n_evaluation':len(split['evaluation'])})
    print(json.dumps({'partition':name,'raw':{key:summarize(value)['raw']['mean'] for key,value in scores.items()}}),flush=True)
