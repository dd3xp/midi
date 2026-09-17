"""Independent explicit Gaussian kernel + symmetric padding + FFT convolution."""
from pathlib import Path
import datetime,hashlib,json,re
import numpy as np
from scipy.signal import fftconvolve
P=Path(__file__).resolve().parent;C=P.parent if (P.parent/'analysis_arrays.npz').exists() else P.parent/'revision_20260915/companion'
r=json.loads((P/'scale_results.json').read_text());tracks=json.loads((C/'tracks.json').read_text());a=np.load(C/'analysis_arrays.npz')
models=list(next(iter(r['scale_summary'].values())));maxerr=0.;checks=0;collected={}
def close(x,y):
    global maxerr,checks
    if x is None or y is None:assert x is None and y is None;return
    d=abs(x-y);maxerr=max(maxerr,d);checks+=1;assert d<1e-8,(x,y,d)
def pearson(x,y):
    if len(x)<3 or np.std(x)<1e-10 or np.std(y)<1e-10:return None
    return float(np.corrcoef(x,y)[0,1])
for i,row in enumerate(tracks):
    pre=row['prefix'];h=row['hop_time'];y=a[pre+'_amp'].astype(float);T=len(y);notes=a[pre+'_notes']
    starts=np.clip((notes[:,0].astype(float)/h).astype(int),0,T);ends=np.clip((notes[:,1].astype(float)/h).astype(int),0,T)
    keep=ends>starts;delta=np.bincount(starts[keep],minlength=T+1)-np.bincount(ends[keep],minlength=T+1);mask=np.cumsum(delta)[:T]>0
    base=np.stack([y]+[a[pre+'_'+m].astype(float) for m in models])
    for key,expect in r['scale_per_track'][i]['conditions'].items():
        dom,eps,sig=re.fullmatch(r'(linear|log)_eps(None|[0-9.e+-]+)_sigma([0-9.]+)',key).groups()
        sig=float(sig)/h;z=base if dom=='linear' else np.log(base+float(eps))
        if sig:
            rad=int(4*sig+.5);x=np.arange(-rad,rad+1,dtype=float);kernel=np.exp(-x*x/(2*sig*sig));kernel/=kernel.sum()
            z=fftconvolve(np.pad(z,((0,0),(rad,rad)),mode='symmetric'),kernel[None,:],mode='valid',axes=1)
        for j,m in enumerate(models,1):
            for view,idx in [('full',slice(None)),('active',mask)]:
                value=pearson(z[j,idx],z[0,idx]);close(value,expect[m][view]);collected.setdefault((key,m,view),[]).append(value)
for (key,m,v),xs in collected.items():
    valid=[x for x in xs if x is not None];s=r['scale_summary'][key][m][v]
    assert len(valid)==s['tracks'];close(float(np.mean(valid)) if valid else None,s['mean'])
for f,h in r['sha256'].items():
    p=P/f if (P/f).exists() else C/f
    assert hashlib.sha256(p.read_bytes()).hexdigest()==h,f
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'passed','numeric_checks':checks,'maximum_error':maxerr,
 'implementation':'explicit normalized Gaussian kernel, NumPy symmetric pad, scipy.signal FFT convolution; original uses scipy.ndimage.gaussian_filter1d',
 'scale_results_sha256':hashlib.sha256((P/'scale_results.json').read_bytes()).hexdigest()}
(P/'scale_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
