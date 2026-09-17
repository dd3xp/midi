# Short-note Within sensitivity

Specified on 15 September 2026 after the v0.28 review, before this calculation. This is a descriptive check of the frozen outputs, not preregistration or independent validation.

Keep original cached intervals, time-to-frame truncation, track weighting and the standard-deviation threshold 1e-10. Compute Within for each of the eight internal predictors at minimum interval lengths 4 (primary), 10, 20 and 50 frames, nominally 40, 100, 200 and 500 ms at 100 Hz. Do not pick a best threshold or replace the primary measure. Report all thresholds, eligible-note counts, defined-note counts and defined-track counts. Shortening the evaluated population changes the estimand; this is not a correction of short-note ground truth.

For each threshold, report the three neural predictors minus both instrument OLS and shared envelope, using tracks where both Within values are defined, equal track weighting and 5,000 whole-group paired bootstrap draws with seed 20260911. This remains conditional on the retained predictions. Check the four-frame values against the original reports. Do not train, change predictions or seek additional models.
