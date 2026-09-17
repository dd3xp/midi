"""Frozen public MIDI-DDSP diagnostic: one complete CPU voice per worker."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', TF_CPP_MIN_LOG_LEVEL='2', OMP_NUM_THREADS='2')
from pathlib import Path
from datetime import datetime, timezone
import sys, json, hashlib, subprocess, time, traceback, fcntl, re, resource
import numpy as np
ROOT = Path('/home/dd3xp/icassp_external_20260912')
RUN = ROOT/'runs/midi_ddsp_public_v1'
INPUT = RUN/'input'
OUT = RUN/'audio'
READ = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
SHA = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)

def seed_for(track, lane):
    return int.from_bytes(hashlib.sha256(f'42|midi-ddsp|{track}|{lane}|full'.encode()).digest()[:4], 'big') % (2**31-1)

def boundary(protocol):
    if (RUN/'STOP').exists(): raise RuntimeError('Remote STOP requested')
    expiry = re.sub(r'(\.\d{6})\d+', r'\1', protocol['execution']['expires_utc'].replace('Z','+00:00'))
    if datetime.now(timezone.utc) >= datetime.fromisoformat(expiry):
        raise RuntimeError('Authorized deadline reached; no new worker')

def worker(key, j):
    protocol = READ(INPUT/'protocol.json')
    boundary(protocol)
    item = next(t for t in protocol['tracks'] if t['key']==key)
    lane = item['timeline']['lanes'][j]
    directory = OUT/key
    def phase(name):
        save(directory/f'lane{j}_status.json', dict(utc=datetime.now(timezone.utc).isoformat(),
             pid=os.getpid(),status=name,track=item['track'],lane=j))
    phase('loading')
    import tensorflow as tf
    import importlib.metadata
    assert tf.__version__=='2.7.0' and importlib.metadata.version('ddsp')=='3.2.0'
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    assert not tf.config.list_physical_devices('GPU')
    sys.path.insert(0, str(ROOT/'repos/midi-ddsp'))
    from midi_ddsp import load_pretrained_model
    from midi_ddsp.utils.inference_utils import expression_generator_output_to_conditioning_df, conditioning_df_to_audio
    synth, expr = load_pretrained_model()
    weights = ROOT/'repos/midi-ddsp/midi_ddsp/midi_ddsp_model_weights_urmp_9_10'
    synth.load_weights(str(weights/'synthesis_generator/50000')).assert_existing_objects_matched()
    expr.load_weights(str(weights/'expression_generator/5000')).assert_existing_objects_matched()
    seed = seed_for(item['track'],j)
    tf.random.set_seed(seed); np.random.seed(seed)
    sequence = dict(note_pitch=tf.constant([[t['pitch'] for t in lane]],tf.int64),
        note_length=tf.constant([[[t['length_frames']/250] for t in lane]],tf.float32),
        instrument_id=tf.constant([item['instrument_id']],tf.int64))
    tick = time.monotonic(); phase('expression')
    expression = expr(sequence,out=None,training=False)
    assert np.isfinite(np.asarray(expression['output'])).all()
    df = expression_generator_output_to_conditioning_df(expression['output'],sequence)
    assert df['onset'].astype(int).tolist()==[t['onset'] for t in lane]
    assert df['offset'].astype(int).tolist()==[t['offset'] for t in lane]
    df.to_csv(directory/f'lane{j}_conditioning.csv',index=False)
    phase('synthesis')
    audio, params, _ = conditioning_df_to_audio(synth,df,sequence['instrument_id'],display_progressbar=False)
    y = np.asarray(audio,dtype=np.float32).reshape(-1)
    assert y.shape==(item['timeline']['synthesis_frames']*64,) and np.isfinite(y).all()
    assert all(np.isfinite(np.asarray(p)).all() for p in params)
    np.save(directory/f'lane{j}_audio.npy',y)
    np.savez_compressed(directory/f'lane{j}_params.npz',**{k:np.asarray(v) for k,v in zip(['f0','amplitudes','harmonic_distribution','noise_magnitudes'],params)})
    files={p.name:SHA(p) for p in [directory/f'lane{j}_audio.npy',directory/f'lane{j}_params.npz',directory/f'lane{j}_conditioning.csv']}
    save(directory/f'lane{j}_complete.json',dict(utc=datetime.now(timezone.utc).isoformat(),
        track=item['track'],lane=j,seed=seed,samples=len(y),finite=True,
        checkpoint_objects_matched=True,seconds=time.monotonic()-tick,
        peak_process_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,files=files))
    phase('complete')

def parent():
    RUN.mkdir(parents=True,exist_ok=True)
    lock=(RUN/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (RUN/'queue_status.json').exists(): raise RuntimeError('Existing attempt; no automatic restart')
    protocol=READ(INPUT/'protocol.json');protocol_sha=SHA(INPUT/'protocol.json')
    assert protocol_sha==(INPUT/'protocol.sha256').read_text().strip()
    boundary(protocol)
    assert SHA(Path(__file__))==protocol['runner_sha256']
    for name,expected in protocol['vendor_files'].items():
        assert SHA(ROOT/'repos/midi-ddsp'/name)==expected, name
    assert SHA(INPUT/'timeline_rms_adapter.py')==protocol['RMS']['helper_sha256']
    sys.path.insert(0,str(INPUT));from timeline_rms_adapter import audio_rms
    state=dict(pid=os.getpid(),status='ready',device='CPU',scope=protocol['identity'],
        protocol_sha256=protocol_sha,total_tracks=186,completed_tracks=0,completed_lanes=0,
        completed_audio_seconds=0,gpu_work_started=False)
    tick=time.monotonic()
    def update(**extra):
        state.update(extra,utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-tick)
        state['audio_seconds_per_wall_second']=state['completed_audio_seconds']/max(state['elapsed_seconds'],1e-6)
        save(RUN/'queue_status.json',state)
        with (RUN/'history.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
    save(RUN/'launch.json',dict(command=[sys.executable,__file__],protocol_sha256=protocol_sha,
        runner_sha256=SHA(Path(__file__)),execution=protocol['execution']))
    update()
    try:
        for item in protocol['tracks']:
            directory=OUT/item['key'];directory.mkdir(parents=True,exist_ok=True)
            mix=None;records=[]
            for j in range(item['timeline']['lane_count']):
                boundary(protocol)
                available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))
                if available<12*1024*1024: raise RuntimeError('CPU memory headroom below 12 GiB; queue paused')
                with (directory/f'lane{j}.log').open('wb') as log:
                    child=subprocess.Popen([sys.executable,__file__,'--worker',item['key'],str(j)],stdout=log,stderr=subprocess.STDOUT)
                    update(status='synthesizing',current_track=item['track'],current_lane=j,worker_pid=child.pid)
                    code=child.wait()
                if code!=0: raise RuntimeError(f'Worker exit {code}; no automatic restart')
                record=READ(directory/f'lane{j}_complete.json')
                assert record['seed']==seed_for(item['track'],j) and record['finite']
                for name,expected in record['files'].items():assert SHA(directory/name)==expected
                y=np.load(directory/f'lane{j}_audio.npy')
                if mix is None: mix=y.copy()
                else: mix+=y
                records.append(record)
                state['completed_lanes']+=1;state['completed_audio_seconds']+=len(y)/16000
                update(status='between_lanes',worker_pid=None)
            assert np.isfinite(mix).all()
            rms=audio_rms(mix,pad_mode='constant',target_frames=item['target_frames'])
            assert rms.shape==(item['target_frames'],) and np.isfinite(rms).all()
            np.save(directory/'mix_audio.npy',mix);np.save(directory/'rms.npy',rms)
            save(directory/'complete.json',dict(utc=datetime.now(timezone.utc).isoformat(),track=item['track'],key=item['key'],
                protocol_sha256=protocol_sha,samples=len(mix),RMS_frames=len(rms),finite=True,lanes=records,
                mix_sha256=SHA(directory/'mix_audio.npy'),rms_sha256=SHA(directory/'rms.npy')))
            del mix,y,rms
            state['completed_tracks']+=1;update(status='between_tracks')
        update(status='complete_collection_pending',worker_pid=None)
    except Exception as exc:
        update(status='paused',error=repr(exc),traceback=traceback.format_exc());raise

if __name__=='__main__':
    try:
        if len(sys.argv)>1 and sys.argv[1]=='--worker':worker(sys.argv[2],int(sys.argv[3]))
        else:parent()
    except Exception:
        if RUN.exists(): (RUN/('worker_error.log' if '--worker' in sys.argv else 'controller_error.log')).write_text(traceback.format_exc())
        raise
