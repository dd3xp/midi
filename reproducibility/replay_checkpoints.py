"""Repeatable CPU inference from retained seed-42 checkpoints; inputs stay immutable."""
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
    ap.add_argument('--all-tracks', action='store_true')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    source = args.execution.resolve()
    for rel, expected in json.loads((source/'manifest.json').read_text()).items():
        assert hashlib.sha256((source/rel).read_bytes()).hexdigest() == expected, rel
    sys.path.insert(0, str(source/'mamba_s4d/code'))
    from train import make_model
    sys.path.insert(0, str(source/'bigru/code'))
    from independent_bigru import IndependentBiGRU
    torch.set_num_threads(2)
    rows = torch.load(source/'data/dataset_corrected.pt', map_location='cpu', weights_only=False)
    folds = json.loads((source/'data/folds.json').read_text())
    results = []
    for name in ['mamba', 's4d', 'bigru']:
        for fold in range(4):
            job = source/f'checkpoints/{name}_fold{fold}'
            model = IndependentBiGRU() if name == 'bigru' else make_model(name)
            model.load_state_dict(torch.load(job/'best.pt', map_location='cpu', weights_only=True), strict=True)
            model.cpu().eval()
            saved = {r['track']: np.asarray(r['amp_pred']) for r in torch.load(job/'predictions.pt', map_location='cpu', weights_only=False)}
            ids = folds[fold]['evaluation']
            assert {rows[i]['track'] for i in ids} == set(saved)
            if not args.all_tracks:
                ids = [min(ids, key=lambda i: (len(rows[i]['amp']), rows[i]['track']))]
            for i in ids:
                row = rows[i]
                started = time.monotonic()
                with torch.inference_mode():
                    prediction = model(torch.from_numpy(row['features']).unsqueeze(0))[1].squeeze(0).numpy()
                expected = saved[row['track']]
                assert prediction.shape == expected.shape and np.isfinite(prediction).all()
                result = dict(model=name, fold=fold, track=row['track'], seconds=time.monotonic()-started,
                    max_abs=float(np.abs(prediction-expected).max()),
                    close=bool(np.allclose(prediction, expected, rtol=1e-3, atol=1e-5)))
                results.append(result)
                print(json.dumps(result), flush=True)
    output = args.output or ROOT/'runs'/('checkpoint_replay_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(dict(torch=torch.__version__, all_tracks=args.all_tracks, results=results), stream, indent=2)
    assert all(r['close'] for r in results), 'Inspect numerical differences in the report; inputs were not modified'

if __name__ == '__main__':
    main()
