# Direct shared-template preservation audit

2026-09-16, after round3 reviews of v0.37 and after all old/new results. This is a retrospective audit of the existing seed42 186-track arrays, not a new confirmatory experiment or a new model fit.

Use every saved Note OLS and shared-envelope output, original annotated spans, stored float32 hop, horizon and minimum four frames. For each eligible note compare the two predicted interval means directly. Report absolute relative difference using max(NoteOLS mean,1e-7), absolute log difference with epsilon1e-7, all-note and frame-nonoverlap counts, median/95th percentile/maximum, and per-track Note/Error difference distributions. Do not choose thresholds after seeing results.

Frame-nonoverlap notes must have no frame shared with any other annotated interval, including shorter notes. Additionally report Full and Note means/differences on all tracks with no overlapping annotated frames at all, preserving their complete original horizons and gaps. This is a defined subset, not a replacement of the original population. Full on a masked note union is not called full-track correlation.

Retain every result, including a materially nonzero preservation error if found. No output is changed. Check interval means by direct sums and cumulative sums. These checks address the control's numerical validity, not whether Note correlations measure perception.
