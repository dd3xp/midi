# Descriptive decomposition of the existing note log error

15 September 2026, after the v0.27 review. Preserve all original targets, predictions, intervals, main correlations and uncalibrated note RMSE. No fitting or calibration is applied to predictions.

For each track/model, let e_n = ln(predicted interval mean +1e-7) - ln(target interval mean +1e-7). Let b=mean(e), c=sqrt(mean((e-b)^2)), and E=sqrt(mean(e^2)). Verify E^2=b^2+c^2. Report equal-track mean signed b, absolute b, c, E, and the fraction sum(b^2)/sum(E^2), explicitly a ratio of sums of per-track mean-squared errors, not a decomposition of mean RMSE or correlation. Display all eight internal predictors.

For all three neural predictors versus both references, within the previously defined Full-up/Error-up subset, report how often b^2 increases, c^2 increases, and each alone or both. Counts describe this fixed subset and do not establish a cause. Report paired mean changes in b^2 and c^2, with no additional significance claims. A signed log offset cannot be attributed specifically to recording gain; the centered residual is not a measure of perceived expression or isolated dynamic-range compression. This explanatory analysis uses no extra labels, model, seed, data split or training run.
