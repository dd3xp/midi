# Review-driven diagnostic revision: fixed analysis plan

Recorded before computing the new sensitivity results. This is a post-review analysis of previously studied data, not preregistration or independent confirmation.

No model training, checkpoint selection, original prediction modification, or split change. Use all 186 cached tracks and the frozen Mamba/S4D/BiGRU and five reference predictions. Preserve original analysis unchanged.

Recompute original four correlation views and verify against the frozen report. Define overlap using all positive-length annotations (including notes shorter than the primary four-frame minimum), before selecting eligible notes. A frame interval is the original half-open integer-truncated interval using cached hop_time. An annotation is excluded as a whole if it touches a frame with multiplicity >1. Independently define overlap on original continuous-time half-open annotations, without rounding; exclude either whole note in every overlapping pair. Do not truncate notes to retain convenient fragments.

Report all-note, frame-nonoverlapping-note and continuous-nonoverlapping-note analyses on the same tracks when metrics are defined. Also report the whole-track subsets with no frame overlap and the existing frozen 95-track native-single-lane population, plus target deduplication and URMP/remaining-corpora subsets. Within-note averaging remains per track, then equal track weight. Undefined correlations remain missing with denominators. These subsets probe annotation/aggregation sensitivity, not isolated-source sound pressure or new generalization.

Report all model means and paired differences for Mamba versus instrument OLS/shared envelope/global OLS/BiGRU/S4D, S4D/BiGRU versus shared envelope/instrument OLS. Use 5,000 piece-group bootstrap draws, seed20260911, track-weighted means and pointwise percentile intervals, conditional on fixed predictions. No interval selection, equivalence claim or multiple-comparison family claim.

Also compute linear and natural-log RMS error on full tracks and original note-union active frames (epsilon1e-7), with no fitted test gain, offset, per-track renormalization or model-dependent masking. These are errors in the cached target units, not calibrated SPL or perceptual quality. Retain every model including the gate and zero-outside-note references; full-track log errors may be strongly affected by their zeros and target tails. Report that behavior explicitly rather than discarding it.

The supplementary package will contain exact metrics, intervals, overlap counts, input/fold manifests, code/configuration, hashes, and reproducibility instructions. Any unresolved provenance of historical target generation must remain explicit. No public upload is implied by creation of a local package.
