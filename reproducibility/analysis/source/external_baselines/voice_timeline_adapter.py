"""Annotation-only candidate: minimum monophonic lanes, no clipped intervals.

This is an explicit input adaptation, not original MIDI-DDSP's mono converter.
It does not create isolated training audio or resolve model training eligibility.
No amplitude targets, predictions, or learned voice assignment are used.
"""
import numpy as np


def partition_timeline(notes, target_frames, fs=250):
    notes = np.asarray(notes, dtype=np.float64)
    if notes.ndim != 2 or notes.shape[1] < 3 or not len(notes):
        raise ValueError('Expected nonempty note array')
    if not np.isfinite(notes[:, :3]).all():
        raise ValueError('Non-finite annotation')
    if np.any(notes[:, 0] < 0) or np.any(notes[:, 1] <= notes[:, 0]):
        raise ValueError('Invalid interval')
    if np.any(notes[:, 2] != np.rint(notes[:, 2])) or np.any((notes[:, 2] < 1) | (notes[:, 2] > 127)):
        raise ValueError('Invalid MIDI pitch')
    if fs not in (250, 100) or target_frames < 1:
        raise ValueError('Expected audited 250/100Hz grid and positive target length')
    on = np.rint(notes[:, 0] * fs).astype(np.int64)
    off = np.rint(notes[:, 1] * fs).astype(np.int64)
    if np.any(off <= on):
        raise ValueError('Quantization collapsed an interval; no silent deletion')
    end = max(int(off.max()) + fs, int(np.ceil((target_frames - 1) * fs / 100)) + 1)
    lanes, ends = [], []
    # Stable ties use source index, never pitch priority, loudness, or model scores.
    for index in sorted(range(len(notes)), key=lambda i: (int(on[i]), i)):
        start, stop = int(on[index]), int(off[index])
        lane = next((j for j, cursor in enumerate(ends) if cursor <= start), len(lanes))
        if lane == len(lanes):
            lanes.append([])
            ends.append(0)
        if start > ends[lane]:
            lanes[lane].append(dict(pitch=0, onset=ends[lane], offset=start, source_note=None))
        lanes[lane].append(dict(pitch=int(notes[index, 2]), onset=start, offset=stop, source_note=index))
        ends[lane] = stop
    for lane, cursor in zip(lanes, ends):
        lane.append(dict(pitch=0, onset=cursor, offset=end, source_note=None))
        for token in lane:
            token['length_frames'] = token['offset'] - token['onset']
    return dict(fs=fs, synthesis_frames=end, lanes=lanes, lane_count=len(lanes),
                notes=len(notes), clipped_notes=0, deleted_notes=0,
                max_endpoint_quantization_seconds=float(np.max(np.abs(np.stack([on, off], axis=1) / fs - notes[:, :2]))))
