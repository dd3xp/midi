"""Exact frozen metric/reference functions, extracted without training imports."""
import numpy as np

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

def intervals(row, minimum=4):
    n, hop = len(row['amp']), row['hop_time']
    for note in row['notes']:
        start = max(0, int(float(note[0]) / hop))
        end = min(n, int(float(note[1]) / hop))
        if end - start >= minimum:
            yield start, end

def gate(row):
    x = np.zeros(len(row['amp']), dtype=float)
    for start, end in intervals(row, 1):
        x[start:end] = 1
    return x

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
