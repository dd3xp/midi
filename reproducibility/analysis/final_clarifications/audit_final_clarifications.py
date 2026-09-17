def assert_reproduced(actual, expected, path='root'):
    """Exact structure/count/hash checks; float-only 1e-12 platform tolerance."""
    import math
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            if key != 'utc': assert_reproduced(actual[key], expected[key], path+'/'+key)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected)): assert_reproduced(a, e, path+'/'+str(i))
    elif isinstance(expected, float):
        assert math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (path, actual, expected)
    else:
        assert actual == expected, (path, actual, expected)

from pathlib import Path
import json,numpy as np,datetime,hashlib
P=Path(__file__).resolve().parent
C=P.parent/'revision_20260915/companion';T=P/'template_preservation.json';M=P/'musicnet/results.json'
if (P.parent/'analysis_arrays.npz').exists():
    C=P.parent;T=C/'template_preservation/template_preservation.json';M=C/'musicnet/results.json'
r=json.loads(M.read_text());look={(x['track'],x['model']):x for x in r['per_track']};ids=sorted({x['track'] for x in r['per_track']});d={}
for m in ['mamba','s4d']:
    rows=[]
    for t in ids:
        f=look[t,m]['Full']-look[t,'instrument_ols']['Full'];n=look[t,m]['Note']-look[t,'instrument_ols']['Note']
        rows.append(dict(track=t,Full_gain=f,Note_change=n))
    d[m]=dict(discordant=[x for x in rows if x['Full_gain']>0 and x['Note_change']<0],above_005=dict(full_up=sum(x['Full_gain']>.05 for x in rows),note_down=sum(x['Full_gain']>.05 and x['Note_change']<-.05 for x in rows)))
a=json.loads(T.read_text());ts=json.loads((C/'tracks.json').read_text());meta={x['track']:x for x in ts}
ss=[x for x in a['per_track'] if x['no_overlapping_frames']];groups=sorted({meta[x['track']]['group'] for x in ss})
v=[[x['Full_shape']-x['Full_ols'] for x in ss if meta[x['track']]['group']==g] for g in groups]
rng=np.random.default_rng(20260911)
boot=[np.mean([val for i in draw for val in v[i]]) for draw in rng.integers(0,len(groups),(5000,len(groups)))]
o=dict(tracks=len(ss),groups=len(groups),Full_gain=float(np.mean([val for vals in v for val in vals])),interval=np.percentile(boot,[2.5,97.5]).tolist(),corpora={})
for c in sorted({meta[x['track']]['dataset'] for x in ss}):
    vals=[x['Full_shape']-x['Full_ols'] for x in ss if meta[x['track']]['dataset']==c]
    o['corpora'][c]=dict(n=len(vals),mean=float(np.mean(vals)))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=sha(P/'ROUND6_AUDIT_PLAN.md'),template_audit_sha256=sha(T),musicnet_results_sha256=sha(M),musicnet=d,original_nonoverlap=o)
target=P/'round6_clarification.json'
if target.exists():
    old=json.loads(target.read_text());assert_reproduced(out, old)
    target=P/'round6_clarification_recomputed.json'
target.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
