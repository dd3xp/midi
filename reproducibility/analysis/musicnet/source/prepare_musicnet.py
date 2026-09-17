"""Full-horizon MusicNet adapter fixed before model outcomes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json,sys
import librosa,numpy as np,soundfile as sf,torch
P=Path(__file__).resolve().parent;D=P/'data_feasibility';R=P.parents[1]/'local_run_20260912'
sys.path.insert(0,str(R/'source_snapshot'))
from src.model.solo.dataset import notes_to_frame_features
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
frozen=json.loads((P/'confirmation_frozen.json').read_text())
assert sha(P/'CONFIRMATION_PLAN.md')==frozen['plan_sha256']
manifest=json.loads((D/'acquisition_manifest.json').read_text())
assert manifest['selection']==frozen['selected_metadata']
for item in manifest['files']:assert sha(D/'selected'/item['path'])==item['sha256']
out=P/'musicnet';out.mkdir(exist_ok=True);assert not (out/'dataset.pt').exists()
rows=[];audits=[]
for meta in frozen['selected_metadata']:
    ident=meta['id'];wav=D/'selected'/f'{ident}.wav';lab=D/'selected'/f'{ident}.csv'
    y,sr=sf.read(wav,dtype='float32',always_2d=True);channels=y.shape[1];samples=len(y)
    labels=list(csv.DictReader(lab.open(encoding='utf-8')))
    assert labels and {'start_time','end_time','instrument','note'}<=set(labels[0])
    inst='vn' if meta['ensemble']=='Solo Violin' else 'vc'
    programs=sorted({int(n['instrument']) for n in labels})
    assert set(programs)<={41,42,43},programs
    src=np.array([[int(n['start_time'])/sr,int(n['end_time'])/sr,int(n['note']),0] for n in labels],dtype=np.float64)
    assert np.isfinite(src).all() and (src[:,0]>=0).all() and (src[:,1]>src[:,0]).all()
    assert src[:,1].max()<=samples/sr,'Annotations outside waveform horizon'
    notes=src.astype(np.float32);mono=y.mean(axis=1)
    audio=librosa.resample(mono,orig_sr=sr,target_sr=16000,res_type='soxr_hq')
    amp=librosa.feature.rms(y=audio,frame_length=320,hop_length=160,center=True,pad_mode='constant')[0]
    hop=float(np.float32(.01))
    ff=notes_to_frame_features(notes,len(amp),hop,n_features=20,instrument=inst)
    vv=notes.copy();vv[:,3]=127
    assert np.array_equal(ff,notes_to_frame_features(vv,len(amp),hop,n_features=20,instrument=inst))
    assert ff.shape==(len(amp),20) and np.isfinite(ff).all() and np.isfinite(amp).all()
    overlaps=sum(bool(np.any((src[:,0]<n[1])&(src[:,1]>n[0])&(np.arange(len(src))!=i))) for i,n in enumerate(src))
    audit=dict(id=ident,work=meta['catalog_name'],instrument=inst,source_sample_rate=sr,
               source_samples=samples,source_channels=channels,resampled_samples=len(audio),frames=len(amp),
               notes=len(notes),label_programs=programs,recording_instrument_from_metadata=inst,
               overlapping_notes=overlaps,simultaneous_extra_onsets=len(src)-len(np.unique(src[:,0])),
               max_float32_time_error_seconds=float(np.abs(src[:,:2]-notes[:,:2]).max()),
               source_start_seconds=float(src[:,0].min()),source_last_offset_seconds=float(src[:,1].max()),
               complete_audio_seconds=samples/sr,wav_sha256=sha(wav),labels_sha256=sha(lab))
    audits.append(audit);print(json.dumps(audit),flush=True)
    rows.append(dict(track=f'MusicNet/{ident}',group=meta['catalog_name'],instrument=inst,
                     amp=amp,features=ff,notes=notes,hop_time=hop,metadata=meta))
torch.save(rows,out/'dataset.pt')
record=dict(utc=datetime.now(timezone.utc).isoformat(),plan_sha256=frozen['plan_sha256'],
            adapter_sha256=sha(Path(__file__)),dataset_sha256=sha(out/'dataset.pt'),
            source_release_md5=manifest['release_md5'],tracks=21,works=5,
            libraries=dict(librosa=librosa.__version__,numpy=np.__version__,soundfile=sf.__version__),
            waveform_metrics_or_model_outputs_inspected=False,audits=audits)
(out/'dataset_manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
