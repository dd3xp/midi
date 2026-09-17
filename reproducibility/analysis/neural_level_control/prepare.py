from pathlib import Path
import collections,datetime,hashlib,json
import numpy as np
P=Path(__file__).resolve().parent;C=P.parent/'revision_20260915/companion'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inputs={name:sha(C/name) for name in ['analysis_arrays.npz','tracks.json','folds.json']}
original=json.loads((C/'manifest.json').read_text())
assert all(original[k]==v for k,v in inputs.items())
ts=json.loads((C/'tracks.json').read_text());folds=json.loads((C/'folds.json').read_text())
assert len(ts)==len({t['track'] for t in ts})==186
evaluated=[]
for fold in folds:
    roles=[{ts[i]['group'] for i in fold[k]} for k in ['train','validation','evaluation']]
    assert all(not roles[i]&roles[j] for i in range(3) for j in range(i))
    evaluated.extend(fold['evaluation'])
assert sorted(evaluated)==list(range(186))
selected=[];a=np.load(C/'analysis_arrays.npz')
for t in ts:
    amp=a[t['prefix']+'_amp']
    # The original grouping hash covers amplitude plus pitch, not amplitude alone.
    assert hashlib.sha256(amp.tobytes()+a[t['prefix']+'_f0'].tobytes()).hexdigest()==t['target_sha256']
    counts=np.zeros(len(amp),int);spans=[]
    for n in a[t['prefix']+'_notes']:
        s=max(0,int(float(n[0])/t['hop_time']));e=min(len(amp),int(float(n[1])/t['hop_time']))
        if e>s:counts[s:e]+=1;spans.append([s,e])
    if counts.max()<=1:
        assert spans
        selected.append(dict(track=t['track'],prefix=t['prefix'],group=t['group'],dataset=t['dataset'],index=t['index'],frames=len(amp),spans=spans))
assert len(selected)==133 and len({t['group'] for t in selected})==34
out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=sha(P/'PLAN.md'),input_sha256=inputs,roles_verified=True,unique_tracks=186,selected_tracks=selected,groups=34)
assert not (P/'input_manifest.json').exists()
(P/'input_manifest.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k!='selected_tracks'},indent=2))
