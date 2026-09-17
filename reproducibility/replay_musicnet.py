"""CPU checkpoint replay on frozen MusicNet features; no fitting or selection."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execution', type=Path, default=ROOT/'assets/execution')
    ap.add_argument('--all-tracks', action='store_true', help='All 21 recordings; default: shortest complete recording for each model')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    execution = args.execution.resolve()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for rel, expected in json.loads((execution/'manifest.json').read_text()).items():
        assert sha(execution/rel) == expected, rel
    analysis = ROOT/'analysis'
    for rel, expected in json.loads((analysis/'manifest.json').read_text()).items():
        assert sha(analysis/rel) == expected, rel
    companion = analysis/'musicnet'
    frozen = json.loads((companion/'confirmation_frozen.json').read_text())
    tracks = json.loads((companion/'tracks.json').read_text())
    assert len(tracks) == len({r['track'] for r in tracks}) == 21
    assert len({r['group'] for r in tracks}) == 5
    if not args.all_tracks:
        tracks = [min(tracks, key=lambda r: (r['frames'], r['track']))]
    inputs = np.load(companion/'inputs.npz')
    saved = np.load(companion/'predictions_all.npz')
    sys.path.insert(0, str(execution/'mamba_s4d/code'))
    from train import make_model
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    results = []
    for name in ['mamba', 's4d']:
        checkpoint = execution/f'checkpoints/{name}_fold0/best.pt'
        assert sha(checkpoint) == frozen['original_inputs'][f'jobs/{name}_fold0/best.pt']
        model = make_model(name).cpu().eval()
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        for row in tracks:
            track = row['track'].split('/')[-1]
            features = inputs[track+'_features']
            assert features.shape == (row['frames'], 20)
            started = time.monotonic()
            with torch.inference_mode():
                prediction = model(torch.from_numpy(features).unsqueeze(0))[1].squeeze(0).numpy()
            expected = saved[track+'_'+name]
            assert prediction.shape == expected.shape and np.isfinite(prediction).all()
            result = dict(model=name, track=row['track'], frames=row['frames'],
                seconds=time.monotonic()-started, max_abs=float(np.abs(prediction-expected).max()),
                close=bool(np.allclose(prediction, expected, rtol=1e-4, atol=1e-6)))
            results.append(result)
            print(json.dumps(result), flush=True)
    output = args.output or ROOT/'runs'/('musicnet_replay_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(dict(torch=torch.__version__, all_tracks=args.all_tracks, results=results), stream, indent=2)
    assert all(r['close'] for r in results), 'CPU replay differs: inspect report; saved predictions were not modified'

if __name__ == '__main__':
    main()
