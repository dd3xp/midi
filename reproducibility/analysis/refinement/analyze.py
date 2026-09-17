"""Additional fixed-output diagnostics, specified in ANALYSIS_PLAN.md."""
from pathlib import Path
import datetime, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parent
C = P.parent if (P.parent/'analysis_arrays.npz').exists() else P.parent / 'revision_20260915/companion'
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
tracks = json.loads((C/'tracks.json').read_text())
old = json.loads((C/'expected_results.json').read_text())
errors = json.loads((C/'level_and_aggregation_audit.json').read_text())['error_per_track']
arrays = np.load(C/'analysis_arrays.npz')
models = list(old['per_track'])
lookup = {m:{v['track']:v['views']['all'] for v in old['per_track'][m]} for m in models}
groups = [r['group'] for r in tracks]

def weights(indices):
    gs = sorted({groups[i] for i in indices})
    ix = [gs.index(groups[i]) for i in indices]
    draws = np.random.RandomState(20260911).randint(len(gs), size=(5000,len(gs)))
    return np.stack([(draws==j).sum(1) for j in range(len(gs))],1)[:,ix]

def stats(values, indices):
    v = np.asarray(values,dtype=float)
    keep = np.isfinite(v)
    indices = [i for i,k in zip(indices,keep) if k]
    v = v[keep]
    w = weights(indices)
    boot = w@v/w.sum(1)
    return dict(mean=float(v.mean()), ci95=np.quantile(boot,[.025,.975]).tolist(),
                tracks=len(indices), groups=len({groups[i] for i in indices}))

def corr(x,y):
    if len(x)<3 or np.std(x)<1e-10 or np.std(y)<1e-10: return None
    return float(np.corrcoef(x,y)[0,1])

def measure(y,p,spans):
    ym = np.array([y[s:e].mean() for s,e in spans])
    pm = np.array([p[s:e].mean() for s,e in spans])
    yp = np.r_[0.,np.cumsum(y)]; pp = np.r_[0.,np.cumsum(p)]
    ym2 = np.array([(yp[e]-yp[s])/(e-s) for s,e in spans])
    pm2 = np.array([(pp[e]-pp[s])/(e-s) for s,e in spans])
    assert np.allclose(ym,ym2,atol=1e-10,rtol=1e-8)
    assert np.allclose(pm,pm2,atol=1e-10,rtol=1e-8)
    cs = [corr(p[s:e],y[s:e]) for s,e in spans]
    valid = [c for c in cs if c is not None]
    return {'note_mean':corr(pm,ym), 'within_note':float(np.mean(valid)) if valid else None,
            'note_log_rmse':float(np.sqrt(np.mean((np.log(pm+1e-7)-np.log(ym+1e-7))**2))),
            'eligible_notes':len(spans),'defined_shape_notes':len(valid)}

out = {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'sha256':{f.name:sha(f) for f in [P/'ANALYSIS_PLAN.md',P/'analyze.py']+
        [C/n for n in ['analysis_arrays.npz','tracks.json','expected_results.json','level_and_aggregation_audit.json']]},
       'directions':{},'boundary_per_track':[]}
allix = list(range(len(tracks)))
for ref in ['instrument_ols','note_shape']:
    rows = []
    for i,r in enumerate(tracks):
        t = r['track']
        rows.append({'index':i,'track':t,'full_gain':lookup['mamba'][t]['raw']-lookup[ref][t]['raw'],
          'note_gain':lookup['mamba'][t]['note_mean']-lookup[ref][t]['note_mean'],
          'error_reduction':errors[ref][t]['note_log_rmse']-errors['mamba'][t]['note_log_rmse']})
    dx = np.array([r['full_gain'] for r in rows]); dy = np.array([r['note_gain'] for r in rows])
    de = np.array([r['error_reduction'] for r in rows])
    counts = {f'full_{sx}_note_{sy}':int(((dx*sx>0)&(dy*sy>0)).sum()) for sx in [1,-1] for sy in [1,-1]}
    counts.update(full_ties=int((dx==0).sum()),note_ties=int((dy==0).sum()),
                  full_up_error_worse=int(((dx>0)&(de<0)).sum()), full_up=int((dx>0).sum()))
    selected = []
    for sign in [-1,1]:
        ix = np.where((dx>0)&(dy*sign>0))[0]
        if not len(ix): continue
        xy = np.c_[dx[ix],dy[ix]]; scale = np.diff(np.quantile(xy,[.25,.75],axis=0),axis=0)[0]
        scale[scale==0]=1
        dist = np.sum(((xy-np.median(xy,axis=0))/scale)**2,axis=1)
        pick = sorted(zip(dist, [tracks[j]['track'] for j in ix],ix))[0][2]
        selected.append(rows[pick]|{'note_direction':sign})
    w=weights(allix); denom=w@(dx>0); numer=w@((dx>0)&(dy<0))
    out['directions'][ref]={'counts':counts,'discordance_all':stats(((dx>0)&(dy<0)).astype(float),allix),
       'discordance_given_full_up':{'mean':float(((dx>0)&(dy<0)).sum()/(dx>0).sum()),
        'ci95':np.quantile(numer[denom>0]/denom[denom>0],[.025,.975]).tolist(),'valid_draws':int((denom>0).sum())},
       'selected_cases':selected,'per_track':rows}

durations=[]; silence=[]; original_maxerr=0.; changed=0; common_count=0
for i,r in enumerate(tracks):
    pre=r['prefix']; y=arrays[pre+'_amp'].astype(float); notes=arrays[pre+'_notes']; h=r['hop_time']; T=len(y)
    onsets=np.unique(notes[:,0]); base=[]; nxt=[]; mask=np.zeros(T,dtype=bool)
    for n in notes:
        s=max(0,int(float(n[0])/h)); e=min(T,int(float(n[1])/h))
        pos=np.searchsorted(onsets,n[0],side='right')
        en=min(T,int(float(onsets[pos] if pos<len(onsets) else n[1])/h))
        base.append((s,e)); nxt.append((s,en))
        if e>s: mask[s:e]=True
        if e-s>=4: durations.append(float(n[1]-n[0]))
    silence.append(float(1-mask.mean()))
    eligible=[(s,e) for s,e in base if e-s>=4]
    both=[j for j,((s,e),(sn,en)) in enumerate(zip(base,nxt)) if e-s>=4 and en-sn>=4]
    common_count+=len(both); changed+=sum(base[j]!=nxt[j] for j in both)
    record={'index':i,'track':r['track'],'dataset':r['dataset'],'original_eligible':len(eligible),
       'next_eligible':sum(e-s>=4 for s,e in nxt),'common_eligible':len(both),
       'changed_common':sum(base[j]!=nxt[j] for j in both),'original':{},'next':{}}
    for m in models:
        pred=arrays[pre+'_'+m].astype(float)
        orig=measure(y,pred,eligible)
        for key,expected in [('note_mean',lookup[m][r['track']]['note_mean']),
                             ('note_log_rmse',errors[m][r['track']]['note_log_rmse'])]:
            if expected is not None:
                original_maxerr=max(original_maxerr,abs(orig[key]-expected)); assert abs(orig[key]-expected)<1e-8
        for name,spans in [('original',base),('next',nxt)]:
            record[name][m]=measure(y,pred,[spans[j] for j in both])
    out['boundary_per_track'].append(record)
    if i%30==0: print('tracks checked',i,flush=True)

subsets={'all186':allix,'URMP136':[i for i,r in enumerate(tracks) if r['dataset']=='URMP'],
         'nonURMP50':[i for i,r in enumerate(tracks) if r['dataset']!='URMP'],
         'nonBach146':[i for i,r in enumerate(tracks) if r['dataset']!='Bach10']}
out['boundary_summary']={}
for name,ix in subsets.items():
    res={}
    for ref in ['instrument_ols','note_shape']:
        res[ref]={}
        for met in ['note_mean','within_note','note_log_rmse']:
            res[ref][met]={}
            for rule in ['original','next']:
                d=[out['boundary_per_track'][i][rule]['mamba'][met]-out['boundary_per_track'][i][rule][ref][met] for i in ix]
                res[ref][met][rule]=stats(d,ix)
            delta=[(out['boundary_per_track'][i]['next']['mamba'][met]-out['boundary_per_track'][i]['next'][ref][met])-
                   (out['boundary_per_track'][i]['original']['mamba'][met]-out['boundary_per_track'][i]['original'][ref][met]) for i in ix]
            res[ref][met]['rule_change']=stats(delta,ix)
    out['boundary_summary'][name]=res
out['context']={'note_duration_seconds_quantiles':dict(zip(['min','q25','median','q75','max'],np.quantile(durations,[0,.25,.5,.75,1]).tolist())),
 'silent_fraction_per_track_quantiles':dict(zip(['min','q25','median','q75','max'],np.quantile(silence,[0,.25,.5,.75,1]).tolist())),
 'note_fraction_duration_below_half_second':float(np.mean(np.array(durations)<.5)),
 'note_fraction_duration_below_one_second':float(np.mean(np.array(durations)<1)),
 'common_notes':common_count,'changed_common_intervals':changed,'original_metrics_max_error':original_maxerr}
(P/'results.json').write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf-8')

style={'font.family':'DejaVu Sans','font.size':9.2,'axes.titlesize':10,'axes.labelsize':9.2,
 'xtick.labelsize':9.2,'ytick.labelsize':9.2,'legend.fontsize':9.2,'pdf.fonttype':42,'svg.fonttype':'none',
 'axes.spines.top':False,'axes.spines.right':False}
with plt.rc_context(style):
    fig,axs=plt.subplots(1,2,figsize=(7.0,3.0),layout='constrained')
    for ax,(ref,title) in zip(axs,[('instrument_ols','Instrument OLS'),('note_shape','Shared envelope')]):
        rows=out['directions'][ref]['per_track']
        for ds,color,mark,label in [(True,'#276b9b','o','URMP'),(False,'#985522','^','Other corpora')]:
            ss=[r for r in rows if (tracks[r['index']]['dataset']=='URMP')==ds]
            ax.scatter([r['full_gain'] for r in ss],[r['note_gain'] for r in ss],s=13,c=color,marker=mark,alpha=.65,label=label)
        ax.axhline(0,color='.4',lw=.7);ax.axvline(0,color='.4',lw=.7)
        ax.set(xlabel='Full correlation gain',ylabel='Note correlation gain',title='Mamba minus '+title)
        ax.legend(loc='lower right',frameon=False)
    limx=[min(ax.get_xlim()[0] for ax in axs),max(ax.get_xlim()[1] for ax in axs)]
    limy=[min(ax.get_ylim()[0] for ax in axs),max(ax.get_ylim()[1] for ax in axs)]
    for ax in axs:ax.set_xlim(limx);ax.set_ylim(limy)
    fig.savefig(P/'track_gains.pdf');fig.savefig(P/'track_gains.png',dpi=180);plt.close(fig)
    cases=[(ref,c) for ref,d in out['directions'].items() for c in d['selected_cases']]
    fig,axs=plt.subplots(len(cases),1,figsize=(7,7.5),layout='constrained')
    for k,(ax,(ref,c)) in enumerate(zip(axs,cases)):
        r=tracks[c['index']];pre=r['prefix'];y=arrays[pre+'_amp'];h=r['hop_time'];T=len(y);mid=T*h/2
        start=max(0,mid-5);end=min(T*h,mid+5);s=int(start/h);e=min(T,int(end/h));t=np.arange(s,e)*h
        ax.plot(t,y[s:e],color='.15',lw=1,label='Recorded target')
        ax.plot(t,arrays[pre+'_'+ref][s:e],color='#985522',ls='--',lw=.95,label='Reference')
        ax.plot(t,arrays[pre+'_mamba'][s:e],color='#276b9b',lw=.95,label='Mamba')
        top=ax.get_ylim()[1];height=top*.05
        spans=[(max(start,float(n[0])),min(end,float(n[1]))-max(start,float(n[0]))) for n in arrays[pre+'_notes'] if n[1]>start and n[0]<end]
        ax.broken_barh(spans,(top*1.02,height),facecolors='.6',edgecolors='white',lw=.4)
        ax.set_ylim(0,top*1.13);ax.set_xlim(start,end);ax.set_ylabel('RMS')
        reference_name={'instrument_ols':'Instrument OLS','note_shape':'Shared envelope'}[ref]
        title=f"({chr(97+k)}) {r['group']} / {r['instrument']} | {reference_name} | Full {c['full_gain']:+.3f}, Note {c['note_gain']:+.3f}"
        ax.set_title(title,loc='left',fontsize=9.2)
        ax.set_xlabel('Time in track (s)')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='outside upper center',ncol=3,frameon=False)
    fig.savefig(P/'rms_cases.pdf');fig.savefig(P/'rms_cases.png',dpi=150);plt.close(fig)
print(json.dumps({k:v for k,v in out.items() if k not in ['boundary_per_track','directions','sha256']},indent=2),flush=True)
