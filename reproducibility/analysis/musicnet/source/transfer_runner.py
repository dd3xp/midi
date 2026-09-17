"""Frozen full-sequence transfer inference, no training and no automatic retry."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, hashlib, json, random, sys, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(p,obj):
    p=Path(p);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--original-root',type=Path,required=True)
    ap.add_argument('--model',choices=['mamba','s4d'],required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--frozen',type=Path,required=True)
    ap.add_argument('--musicnet',type=Path)
    args=ap.parse_args();out=args.output
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'status.json').exists(),'Do not automatically restart an existing GPU job'
    def status(state,**kw):
        record=dict(utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),state=state,model=args.model,**kw)
        save(out/'status.json',record);print(json.dumps(record),flush=True)
    try:
        frozen=json.loads(args.frozen.read_text(encoding='utf-8'))
        for rel,h in frozen['original_inputs'].items():
            if rel.startswith('baselines/') or (rel.startswith('jobs/') and not rel.startswith(f'jobs/{args.model}_fold0/')):continue
            assert sha(args.original_root/rel)==h,rel
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        torch.cuda.set_per_process_memory_fraction(.80,0);torch.set_num_threads(4)
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        torch.use_deterministic_algorithms(True,warn_only=True)
        sys.path.insert(0,str(args.original_root/'code'))
        from train import make_model
        model=make_model(args.model).cuda().eval()
        checkpoint=args.original_root/f'jobs/{args.model}_fold0/best.pt'
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
        old=torch.load(args.original_root/'artifacts/dataset_corrected.pt',map_location='cpu',weights_only=False)
        split=json.loads((args.original_root/'artifacts/folds.json').read_text())[0]
        audit=dict(python=sys.version,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
                   allocator_fraction=.80,checkpoint_sha256=sha(checkpoint),runner_sha256=sha(__file__),
                   plan_sha256=frozen['plan_sha256'],precision='float32',seed=42,
                   inference='batch one, full sequence, no truncation or gain fitting')
        save(out/'environment.json',audit)
        if args.musicnet:
            assert (args.musicnet.parent/'selection.json').exists(),'Freeze validation selection before new outcomes'
            data_manifest=json.loads((args.musicnet.parent/'dataset_manifest.json').read_text())
            assert sha(args.musicnet)==data_manifest['dataset_sha256']
            rows=torch.load(args.musicnet,map_location='cpu',weights_only=False)
            assert len(rows)==21 and len({r['track'] for r in rows})==21
            # Same audited memory-efficient attention backend as validation feasibility.
            torch.backends.mha.set_fastpath_enabled(False)
            save(out/'input.json',dict(dataset_sha256=sha(args.musicnet),selection_sha256=sha(args.musicnet.parent/'selection.json')))
        else:
            status('short_backend_equivalence')
            source=old[split['train'][0]]['features']
            x=torch.from_numpy(source[:512]).unsqueeze(0)
            with torch.inference_mode():
                # CPU float32 reference avoids the native GPU fused fastpath's reduced precision.
                model=model.cpu();torch.backends.mha.set_fastpath_enabled(True);a=model(x)[1]
                model=model.cuda();x=x.cuda();torch.backends.mha.set_fastpath_enabled(False);b=model(x)[1].cpu()
            delta=float((a-b).abs().max());assert torch.allclose(a,b,rtol=1e-4,atol=1e-6),delta
            del x,a,b
            status('long_training_input_feasibility',frames=34000,backend_max_abs_difference=delta)
            tiled=np.tile(source,(int(np.ceil(34000/len(source))),1))[:34000]
            tick=time.monotonic()
            with torch.inference_mode():
                x=torch.from_numpy(tiled).unsqueeze(0).cuda();p=model(x)[1]
                assert p.shape[-1]==34000 and torch.isfinite(p).all() and (p>=0).all()
            torch.cuda.synchronize()
            save(out/'feasibility.json',dict(frames=34000,seconds=time.monotonic()-tick,
                  peak_gpu_bytes=torch.cuda.max_memory_allocated(),backend_max_abs_difference=delta,
                  backend='torch.backends.mha fastpath disabled; full attention, float32',
                  training_source=old[split['train'][0]]['track']))
            del x,p;torch.cuda.empty_cache()
            rows=[old[i] for i in split['validation']]
        preds=[];timings=[];started=time.monotonic()
        for i,row in enumerate(rows):
            status('inference',completed=i,total=len(rows),track=row['track'])
            tick=time.monotonic()
            with torch.inference_mode():
                x=torch.from_numpy(row['features']).unsqueeze(0).cuda()
                p=model(x)[1].squeeze(0).cpu().numpy().copy()
            assert p.shape==row['amp'].shape and np.isfinite(p).all() and (p>=0).all()
            preds.append(dict(track=row['track'],amp_pred=p))
            timings.append(dict(track=row['track'],frames=len(p),seconds=time.monotonic()-tick))
            torch.save(preds,out/'predictions.partial.pt')
            del x;torch.cuda.empty_cache()
        (out/'predictions.partial.pt').replace(out/'predictions.pt')
        save(out/'timings.json',timings)
        status('complete',completed=len(rows),total=len(rows),seconds=time.monotonic()-started,
               frames_per_second=sum(t['frames'] for t in timings)/sum(t['seconds'] for t in timings),
               prediction_sha256=sha(out/'predictions.pt'),peak_gpu_bytes=torch.cuda.max_memory_allocated())
    except BaseException as e:
        status('failed_no_automatic_restart',error=repr(e));raise

if __name__=='__main__':main()
