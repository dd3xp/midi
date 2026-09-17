"""Shared protocol and diagnostics for the paper-1 supplement."""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d

RUN = Path(__file__).resolve().parent.parent
ROOT = RUN.parent.parent
ART = RUN / 'artifacts'
SCALES = (0.05, 0.1, 0.2, 0.5, 1.0)

def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temp.replace(path)

def corr(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return None
    return float(np.corrcoef(x, y)[0, 1])

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

def metrics(pred, row, sensitivities=True):
    pred, gt = np.asarray(pred, dtype=float), np.asarray(row['amp'], dtype=float)
    assert pred.shape == gt.shape and np.isfinite(pred).all() and (pred >= 0).all()
    active = gate(row).astype(bool)
    result = {'raw': corr(pred, gt), 'active_raw': corr(pred[active], gt[active]),
              'rmse_log': float(np.sqrt(np.mean((np.log(pred+1e-7)-np.log(gt+1e-7))**2)))}
    for minimum in ([2, 4, 8] if sensitivities else [4]):
        means_p, means_g, shapes = [], [], []
        eligible = 0
        for start, end in intervals(row, minimum):
            p, g = pred[start:end], gt[start:end]
            means_p.append(p.mean()); means_g.append(g.mean()); eligible += 1
            value = corr(p, g)
            if value is not None: shapes.append(value)
        suffix = '' if minimum == 4 else f'_min{minimum}'
        result['note_mean'+suffix] = corr(means_p, means_g)
        result['within_note'+suffix] = float(np.mean(shapes)) if shapes else None
        result['eligible_notes'+suffix] = eligible
        result['defined_shape_notes'+suffix] = len(shapes)
    for floor in ([1e-8, 1e-7, 1e-6] if sensitivities else [1e-7]):
        lp, lg = np.log(pred+floor), np.log(gt+floor)
        suffix = '' if floor == 1e-7 else f'_eps{floor:g}'
        result['log_corr'+suffix] = corr(lp, lg)
        for sigma in (SCALES if floor == 1e-7 else [1.0]):
            sf = sigma / row['hop_time']
            result[f'log_smooth_{sigma:g}s'+suffix] = corr(gaussian_filter1d(lp, sf), gaussian_filter1d(lg, sf))
    for sigma in SCALES:
        sf = sigma / row['hop_time']
        result[f'linear_smooth_{sigma:g}s'] = corr(gaussian_filter1d(pred, sf), gaussian_filter1d(gt, sf))
    return result

def summarize(rows):
    output = {}
    for key in rows[0]['metrics']:
        vals = [r['metrics'][key] for r in rows if r['metrics'][key] is not None]
        output[key] = {'mean': float(np.mean(vals)) if vals else None, 'n_tracks': len(vals)}
    return output
