"""Explicit MIDI2Params boundary adapter for already partitioned voices."""
import numpy as np

def frame_inputs(lane, synthesis_frames):
    pitch=np.zeros(synthesis_frames,np.float32)
    onset=np.zeros_like(pitch);offset=np.zeros_like(pitch)
    for token in lane:
        if token['source_note'] is None:continue
        a,b=token['onset'],token['offset']
        assert 0<=a<b<synthesis_frames
        assert not np.any(pitch[a:b])
        pitch[a:b]=token['pitch']
        onset[a]=1
        offset[b]=1  # Original MIDI2Params boundary convention, not end-1.
    return dict(pitches=pitch,onset_arr=onset,offset_arr=offset)
