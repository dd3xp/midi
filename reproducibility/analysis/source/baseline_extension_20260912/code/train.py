"""Single-GPU, resumable training; outer evaluation is never used for selection."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import sys
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from common import RUN, ART, save_json, metrics, summarize

sys.path.insert(0, str(RUN/'source_snapshot'))
from src.model.solo.baseline import BaselineModel, compute_baseline_loss

stop_requested = False
def stop_handler(signum, frame):
    global stop_requested
    stop_requested = True
signal.signal(signal.SIGTERM, stop_handler)
signal.signal(signal.SIGINT, stop_handler)

def atomic_torch_save(path, value):
    temp=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temp);temp.replace(path)

class Crops(Dataset):
    def __init__(self, rows, ids, training, frames=512):
        self.rows, self.training, self.frames = rows,training,frames
        self.entries=[]
        for i in ids:
            if training:self.entries.append((i,None))
            else:
                maximum=max(0,len(rows[i]['amp'])-frames)
                # Exactly eight fixed windows per track; equal validation-track weight.
                self.entries.extend((i,int(start)) for start in np.linspace(0,maximum,8))
    def __len__(self):return len(self.entries)
    def __getitem__(self,index):
        i,start=self.entries[index]; row=self.rows[i]; n=len(row['amp'])
        if start is None:start=np.random.randint(0,max(0,n-self.frames)+1)
        output={}
        for key in ['features','amp','f0','f0_bins']:
            x=row[key][start:start+self.frames]
            if len(x)<self.frames:
                padding=[(0,self.frames-len(x))]+[(0,0)]*(x.ndim-1)
                x=np.pad(x,padding,constant_values=80 if key=='f0_bins' else 0)
            output[key]=torch.from_numpy(np.array(x,copy=True))
        return output

def make_model(name):
    from independent_bigru import IndependentBiGRU
    assert name == 'bigru'
    return IndependentBiGRU()


def loss_for(model,batch,device,loss_kwargs):
    batch={k:v.to(device) for k,v in batch.items()}
    output=model(batch['features'])
    f0,pred=output[:2]; extra=output[2] if len(output)==3 else None
    return compute_baseline_loss(f0,pred,batch['f0_bins'],batch['f0'],batch['amp'],
                                 frame_features=batch['features'],note_info=extra,**loss_kwargs)[0]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['bigru'],required=True)
    parser.add_argument('--fold',type=int,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    assert torch.cuda.is_available() and torch.cuda.device_count()==1, 'Exactly one visible GPU required'
    # Personal 8-GiB device: retain headroom for the desktop.
    torch.cuda.set_per_process_memory_fraction(0.80, device=0)
    torch.set_num_threads(4)
    protocol=json.loads((ART/'protocol.json').read_text())
    folds=json.loads((ART/'folds.json').read_text());fold=folds[args.fold]
    rows=torch.load(ART/'dataset_corrected.pt',map_location='cpu',weights_only=False)
    seed=protocol['model_seed']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    device=torch.device('cuda:0');model=make_model(args.model).to(device)
    smoke_suffix = '_smoke_' + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:8]
    job_name=f'{args.model}_fold{args.fold}'+(smoke_suffix if args.smoke else '')
    job=RUN/'jobs'/job_name;job.mkdir(exist_ok=True)
    generator=torch.Generator().manual_seed(seed)
    train_loader=DataLoader(Crops(rows,fold['train'],True),batch_size=32,shuffle=True,
                            drop_last=False,num_workers=0,generator=generator)
    validation_loader=DataLoader(Crops(rows,fold['validation'],False),batch_size=32,
                                 shuffle=False,num_workers=0)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    epochs=protocol['max_epochs'];patience=protocol['early_stopping_patience']
    def lr_factor(step):
        if step<10:return (step+1)/10
        return max(.001,.5*(1+math.cos(math.pi*(step-10)/max(1,epochs-10))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_factor)
    protocol_hash=hashlib.sha256((ART/'protocol.json').read_bytes()+(ART/'folds.json').read_bytes()).hexdigest()
    code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    history=[];best=float('inf');bad=0;start_epoch=1
    if (job/'last.pt').exists():
        state=torch.load(job/'last.pt',map_location='cpu',weights_only=False)
        assert state['protocol_sha256']==protocol_hash
        migrations_path = ART/'runtime_code_migrations.json'
        migrations = json.loads(migrations_path.read_text()) if migrations_path.exists() else []
        runtime_compatible = any(m['old_sha256']==state['code_sha256'] and m['new_sha256']==code_hash for m in migrations)
        assert state['code_sha256']==code_hash or runtime_compatible, 'Training code changed without an audited runtime-only migration'
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        atomic_torch_save(job/'best.pt', state['best_model'])
        history=state['history'];best=state['best'];bad=state['bad'];start_epoch=state['epoch']+1
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng']);generator.set_state(state['loader_rng'])
        print(f'Resumed {job_name} from completed epoch {start_epoch-1}',flush=True)
    metadata={'job':job_name,'parameters':sum(p.numel() for p in model.parameters()),'protocol_sha256':protocol_hash,
              'code_sha256':code_hash,'gpu':torch.cuda.get_device_name(0),'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
              'split_counts':{k:len(fold[k]) for k in ['train','validation','evaluation']},'seed':seed,
              'selection':'minimum fixed-window validation loss','loss':protocol['loss'],
              'allocator_fraction_limit':0.80, 'torch_version':torch.__version__, 'python_version':sys.version}
    save_json(job/'config.json',metadata)
    print(json.dumps(metadata),flush=True)
    if (job/'complete.json').exists():print('Already complete');return
    wall=time.monotonic()
    for epoch in range(start_epoch,epochs+1):
        if bad>=patience:break
        tick=time.monotonic();model.train();training=[]
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss=loss_for(model,batch,device,protocol['loss'])
            assert torch.isfinite(loss),'Non-finite training loss'
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            training.append((float(loss.detach()),len(batch['amp'])))
            if args.smoke:break
        model.eval();validation=[]
        with torch.inference_mode():
            for batch in validation_loader:
                loss=loss_for(model,batch,device,protocol['loss'])
                assert torch.isfinite(loss),'Non-finite validation loss'
                validation.append((float(loss),len(batch['amp'])))
                if args.smoke:break
        train_loss=sum(x*n for x,n in training)/sum(n for x,n in training)
        val_loss=sum(x*n for x,n in validation)/sum(n for x,n in validation)
        if val_loss<best:
            best=val_loss;bad=0;atomic_torch_save(job/'best.pt',model.state_dict())
        else:bad+=1
        scheduler.step()
        record={'epoch':epoch,'train_loss':train_loss,'validation_loss':val_loss,'best_validation_loss':best,
                'seconds':time.monotonic()-tick,'lr':optimizer.param_groups[0]['lr'],'no_improvement_epochs':bad,
                'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30}
        history.append(record)
        atomic_torch_save(job/'last.pt',{'model':model.state_dict(),'optimizer':optimizer.state_dict(),
          'scheduler':scheduler.state_dict(),'epoch':epoch,'history':history,'best':best,'bad':bad,
          'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
          'cuda_rng':torch.cuda.get_rng_state(),'loader_rng':generator.get_state(),
          'protocol_sha256':protocol_hash,'code_sha256':code_hash,
          'best_model':torch.load(job/'best.pt',map_location='cpu',weights_only=True)})
        save_json(job/'history.json',history);save_json(job/'status.json',{'status':'training',**record})
        print(json.dumps(record),flush=True)
        if args.smoke:
            # Check full-track memory on training data, without looking at outer evaluation.
            longest=max(fold['train'],key=lambda i:len(rows[i]['amp']))
            full_start=time.monotonic()
            with torch.inference_mode():
                full=model(torch.from_numpy(rows[longest]['features']).unsqueeze(0).to(device))[1]
                assert full.shape[-1]==len(rows[longest]['amp']) and torch.isfinite(full).all()
            record['full_train_track_frames']=len(rows[longest]['amp'])
            record['full_train_track_seconds']=time.monotonic()-full_start
            record['peak_gpu_gib_after_full_track']=torch.cuda.max_memory_allocated()/2**30
            save_json(job/'smoke_complete.json',record);print(json.dumps(record),flush=True);return
        if stop_requested or (RUN/'STOP').exists():
            save_json(job/'status.json',{'status':'paused_after_epoch',**record});raise SystemExit(75)
    model.load_state_dict(torch.load(job/'best.pt',map_location=device,weights_only=True));model.eval()
    save_json(job/'status.json',{'status':'evaluating','completed_epochs':len(history)})
    predictions=[];results=[]
    with torch.inference_mode():
        for i in fold['evaluation']:
            row=rows[i];ff=torch.from_numpy(row['features']).unsqueeze(0).to(device)
            pred=model(ff)[1].squeeze(0).cpu().numpy()
            predictions.append({'track':row['track'],'amp_pred':pred.astype(np.float32)})
            results.append({'track':row['track'],'group':row['group'],'instrument':row['instrument'],'metrics':metrics(pred,row)})
            print(f'Evaluated {row["track"]}',flush=True)
            if stop_requested or (RUN/'STOP').exists():raise SystemExit(75)
    atomic_torch_save(job/'predictions.pt',predictions)
    save_json(job/'metrics.json',{'summary':summarize(results),'per_track':results})
    finished={'status':'complete','completed_epochs':len(history),'best_epoch':min(history,key=lambda x:x['validation_loss'])['epoch'],
              'elapsed_this_invocation_seconds':time.monotonic()-wall,'n_evaluation_tracks':len(results)}
    save_json(job/'complete.json',finished);save_json(job/'status.json',finished)
    print(json.dumps(finished),flush=True)

if __name__=='__main__':main()
