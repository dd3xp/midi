"""Independently verify the saved control arrays and reported bootstrap results."""
from pathlib import Path
import datetime,hashlib,json
import numpy as np
P=Path(__file__).resolve().parent;C=P.parent/'revision_20260915/companion'
r=json.loads((P/'results.json').read_text());meta=json.loads((P/'input_manifest.json').read_text())
source=np.load(C/'analysis_arrays.npz');saved=np.load(P/'controlled_predictions.npz')
assert hashlib.sha256((P/'controlled_predictions.npz').read_bytes()).hexdigest()==r['controlled_arrays_sha256']
def pearson(x,y):
    xx=x-x.mean();yy=y-y.mean()
    return float(np.einsum('i,i->',xx,yy)/np.sqrt(np.einsum('i,i->',xx,xx)*np.einsum('i,i->',yy,yy)))
report={};checked=0
for model in ['mamba','s4d']:
    rows=[row for row in r['per_track'] if row['model']==model];lookup={row['track']:row for row in rows};diffs={}
    for t in meta['selected_tracks']:
        pre=t['prefix'];flat=saved[pre+'_'+model+'_flat'];shaped=saved[pre+'_'+model+'_shared'];original=source[pre+'_'+model].astype(float);target=source[pre+'_amp'].astype(float)
        mask=np.zeros(len(target),bool)
        for start,end in t['spans']:
            mask[start:end]=True
            # Direct interval integrals check conservation independently of the renderer's normalized template.
            assert np.isclose(flat[start:end].sum(),original[start:end].sum(),rtol=1e-12,atol=1e-14)
            assert np.isclose(shaped[start:end].sum(),original[start:end].sum(),rtol=1e-12,atol=1e-14)
        assert np.array_equal(flat[~mask],original[~mask]) and np.array_equal(shaped[~mask],original[~mask])
        delta=pearson(shaped,target)-pearson(flat,target)
        assert abs(delta-lookup[t['track']]['Full_difference'])<1e-12
        diffs.setdefault(t['group'],[]).append(delta);checked+=1
    groups=sorted(diffs);rng=np.random.default_rng(20260918)
    boot=np.array([np.mean(np.concatenate([diffs[groups[i]] for i in draw])) for draw in rng.integers(0,len(groups),(5000,len(groups)))])
    interval=np.percentile(boot,[2.5,97.5]);expected=r['summary'][model]['Full_shared_minus_flat']
    assert np.allclose(interval,expected['paired_group_interval'],rtol=0,atol=1e-12)
    report[model]=dict(tracks=len(rows),groups=len(groups),independent_interval=interval.tolist())
out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='passed',saved_model_track_arrays_checked=checked,checks=['interval sum conservation','unchanged gaps and horizon','independent centered-dot-product Pearson','independent explicit-concatenation group bootstrap'],models=report)
(P/'verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
