from pathlib import Path
import datetime,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;C=P.parent if (P.parent/'analysis_arrays.npz').exists() else P.parent/'revision_20260915/companion'
meta=json.loads((C/'tracks.json').read_text());folds=json.loads((C/'folds.json').read_text());r=json.loads((C/'expected_results.json').read_text());e=json.loads((C/'level_and_aggregation_audit.json').read_text())['error_per_track']
look={m:{x['track']:x['views']['all'] for x in xs} for m,xs in r['per_track'].items()}
gs=sorted({x['group'] for x in meta});gix=[gs.index(x['group']) for x in meta];draw=np.random.RandomState(20260911).randint(len(gs),size=(5000,len(gs)));w=np.stack([(draw==i).sum(1) for i in range(len(gs))],1)[:,gix]
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'comparisons':{},
'sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [P/'ALL_PREDICTOR_PLAN.md',P/'all_predictor_directions.py',P/'results.json',C/'analysis_arrays.npz',C/'tracks.json',C/'folds.json',C/'expected_results.json',C/'level_and_aggregation_audit.json']}}
for m in ['mamba','s4d','bigru']:
    for ref in ['instrument_ols','note_shape']:
        dx=np.array([look[m][x['track']]['raw']-look[ref][x['track']]['raw'] for x in meta]);dy=np.array([look[m][x['track']]['note_mean']-look[ref][x['track']]['note_mean'] for x in meta]);de=np.array([e[m][x['track']]['note_log_rmse']-e[ref][x['track']]['note_log_rmse'] for x in meta])
        full=dx>0;op=full&(dy<0);bad=full&(de>0);den=w@full
        d={'full_improved':int(full.sum()),'note_decreased':int(op.sum()),'error_increased':int(bad.sum()),'all_tracks':186,
          'note_loss_quartiles':np.quantile(-dy[op],[.25,.5,.75]).tolist(),'error_increase_quartiles':np.quantile(de[bad],[.25,.5,.75]).tolist(),
          'full_gain_quartiles_when_note_down':np.quantile(dx[op],[.25,.5,.75]).tolist(),
          'quadrants':{f'full_{sx}_note_{sy}':int(((dx*sx>0)&(dy*sy>0)).sum()) for sx in [1,-1] for sy in [1,-1]},
          'full_ties':int((dx==0).sum()),'note_ties':int((dy==0).sum()),'error_ties':int((de==0).sum()),
          'cutoffs':[{'cutoff':c,'full_exceeds':int((dx>c).sum()),'note_opposes':int(((dx>c)&(dy<-c)).sum()),'error_opposes':int(((dx>c)&(de>c)).sum())} for c in [0,.01,.025,.05,.1]]}
        for name,mask in [('note',op),('error',bad)]:
            d[name+'_conditional']={'proportion':float(mask.sum()/full.sum()),'ci95':np.quantile((w@mask)[den>0]/den[den>0],[.025,.975]).tolist()}
        out['comparisons'][m+' minus '+ref]=d
out['composition']={'instruments':{name:sum(x['instrument']==name for x in meta) for name in sorted({x['instrument'] for x in meta})},
 'corpora':{name:{'tracks':sum(x['dataset']==name for x in meta),'groups':len({x['group'] for x in meta if x['dataset']==name})} for name in sorted({x['dataset'] for x in meta})},'folds':folds}
(P/'all_predictor_results.json').write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk not in ['cutoffs','quadrants']} for k,v in out['comparisons'].items()},indent=2))
a=np.load(C/'analysis_arrays.npz');chosen=json.loads((P/'results.json').read_text())['directions'];names={'instrument_ols':'Instrument OLS','note_shape':'Shared envelope'}
with plt.rc_context({'font.family':'DejaVu Sans','font.size':9.2,'axes.labelsize':9.2,'xtick.labelsize':9.2,'ytick.labelsize':9.2,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False}):
    fig,axs=plt.subplots(2,2,figsize=(7,5.2),layout='constrained')
    for ax,(ref,case) in zip(axs.flat,[(ref,x) for ref,d in chosen.items() for x in d['selected_cases']]):
        row=meta[case['index']];pre=row['prefix'];y=a[pre+'_amp'];h=row['hop_time'];spans=[(max(0,int(float(n[0])/h)),min(len(y),int(float(n[1])/h))) for n in a[pre+'_notes']];spans=[(s,e) for s,e in spans if e-s>=4]
        ym=np.array([y[s:e].mean() for s,e in spans]);hi=ym.max()
        for m,col,marker,label in [('mamba','#276b9b','o','Mamba'),(ref,'#985522','^',names[ref])]:
            pm=np.array([a[pre+'_'+m][s:e].mean() for s,e in spans]);hi=max(hi,pm.max())
            ax.scatter(ym,pm,s=10,c=col,marker=marker,alpha=.65,label=label)
        ax.plot([0,hi*1.03],[0,hi*1.03],color='.5',ls=':',lw=.8)
        ax.set(xlim=(0,hi*1.04),ylim=(0,hi*1.04),xlabel='Recorded note-mean RMS',ylabel='Predicted note-mean RMS')
        ax.set_title(f"{row['group']} / {row['instrument']} | {len(spans)} notes",fontsize=9.2,loc='left');ax.legend(frameon=False,fontsize=9.2,loc='upper left')
    fig.savefig(P/'note_mean_cases.pdf');fig.savefig(P/'note_mean_cases.png',dpi=150)
