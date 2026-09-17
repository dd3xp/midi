"""Independent NumPy-only re-evaluation, including row-weight bootstrap."""
from pathlib import Path
import json,hashlib
import numpy as np
P=Path(__file__).resolve().parent
expected=json.loads((P/'expected_results.json').read_text())
tracks=json.loads((P/'tracks.json').read_text())
data=np.load(P/'analysis_arrays.npz',allow_pickle=False)
models=list(expected['per_track']);computed={m:{} for m in models}
def corr(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    if len(a)<3 or a.std()<1e-10 or b.std()<1e-10:return None
    a=a-a.mean();b=b-b.mean()
    return float(np.dot(a,b)/np.sqrt(np.dot(a,a)*np.dot(b,b)))
maxerr=0.; checks=0
for r in tracks:
    t=r['track'];pre=r['prefix'];y=data[pre+'_amp'].astype(float);notes=data[pre+'_notes'];n=len(y)
    starts=np.maximum(0,(notes[:,0].astype(float)/r['hop_time']).astype(int))
    ends=np.minimum(n,(notes[:,1].astype(float)/r['hop_time']).astype(int))
    valid=ends>starts
    delta=np.zeros(n+1,int);np.add.at(delta,starts[valid],1);np.add.at(delta,ends[valid],-1)
    counts=np.cumsum(delta[:-1]);prefix=np.r_[0,np.cumsum(counts>1)]
    base=np.flatnonzero(ends-starts>=4)
    fk=[i for i in base if prefix[ends[i]]==prefix[starts[i]]]
    order=np.argsort(notes[:,0],kind='stable');left=notes[order,0].astype(float);right=notes[order,1].astype(float)
    assert np.all(right>left)
    prior_end=np.r_[-np.inf,np.maximum.accumulate(right)[:-1]]
    following_start=np.r_[left[1:],np.inf]
    bad=set(order[(prior_end>left)|(following_start<right)])
    ck=[i for i in base if i not in bad]
    g=expected['geometry'][t]
    assert (len(base),len(fk),len(ck),int((counts>1).sum()))==(g['eligible_notes'],g['nonoverlap_frame_notes'],g['nonoverlap_continuous_notes'],g['overlap_frames'])
    active=counts>0
    for m in models:
        pred=data[pre+'_'+m].astype(float)
        common={'raw':corr(pred,y),'active_raw':corr(pred[active],y[active])}
        for tag,mask in [('',np.ones(n,bool)),('_active',active)]:
            common['rmse_linear'+tag]=float(np.sqrt(np.mean(np.square(pred[mask]-y[mask]))))
            common['rmse_log'+tag]=float(np.sqrt(np.mean(np.square(np.log(pred[mask]+1e-7)-np.log(y[mask]+1e-7)))))
        views={}
        for name,ix in [('all',base),('frame_nonoverlap',fk),('continuous_nonoverlap',ck)]:
            meansp=[pred[starts[i]:ends[i]].mean() for i in ix];meansy=[y[starts[i]:ends[i]].mean() for i in ix]
            shape=[corr(pred[starts[i]:ends[i]],y[starts[i]:ends[i]]) for i in ix];shape=[v for v in shape if v is not None]
            views[name]={**common,'note_mean':corr(meansp,meansy),'within_note':float(np.mean(shape)) if shape else None,'eligible_notes':len(ix),'defined_shape_notes':len(shape)}
        computed[m][t]=views
        old=next(v['views'] for v in expected['per_track'][m] if v['track']==t)
        for name in views:
            for k,v in views[name].items():
                ref=old[name][k];assert (v is None)==(ref is None)
                if v is not None:
                    err=abs(v-ref);maxerr=max(maxerr,err);assert err<1e-8,(t,m,k,err)
                checks+=1
groups={r['track']:r['group'] for r in tracks}
paired_checks=0
for label,s in expected['subsets'].items():
    view=s['view']
    for pair,metrics in s['paired'].items():
        a,b=pair.split(' minus ')
        for metric,result in metrics.items():
            ts=[t for t in s['tracks'] if computed[a][t][view][metric] is not None and computed[b][t][view][metric] is not None]
            if not ts:assert result is None;continue
            gn=sorted({groups[t] for t in ts});d=np.array([computed[a][t][view][metric]-computed[b][t][view][metric] for t in ts])
            draw=np.random.RandomState(20260911).randint(len(gn),size=(5000,len(gn)))
            groupweights=np.stack([(draw==j).sum(1) for j in range(len(gn))],axis=1)
            weights=groupweights[:,[gn.index(groups[t]) for t in ts]]
            vals=weights@d/weights.sum(1);ci=np.quantile(vals,[.025,.975])
            err=max(abs(d.mean()-result['difference']),float(np.max(np.abs(ci-result['ci95']))));maxerr=max(maxerr,err)
            assert err<1e-8,(label,pair,metric,err)
            assert result['tracks']==len(ts) and result['groups']==len(gn)
            paired_checks+=1
folds=json.loads((P/'folds.json').read_text())
seen=[]
for fold in folds:
    roles=[set(fold[k]) for k in ('train','validation','evaluation')]
    assert sum(map(len,roles))==186 and len(set.union(*roles))==186
    for i in range(3):
        for j in range(i+1,3):assert not {tracks[k]['group'] for k in roles[i]}&{tracks[k]['group'] for k in roles[j]}
    seen.extend(fold['evaluation'])
assert sorted(seen)==list(range(186))
report={'status':'passed','metric_checks':checks,'paired_bootstrap_checks':paired_checks,'max_error':maxerr,'four_fold_group_roles_disjoint':True,'numpy_version':np.__version__}
(P/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
