# Frozen plan: shape control at neural-predicted note levels

This plan is written before computing the new comparisons. It responds to the v0.44.23 independent review and the user's authorization to use the extended submission period for necessary supplementary experiments. This is an additional retrospective diagnostic, not a preregistration or a new independent validation population.

## Question and scope

Does the existing shape-control result depend on taking note levels from global Note OLS? Hold the note means of the already trained Mamba and S4D predictors fixed and vary only within-note shape. Use seed42 archived out-of-fold predictions; no training, model selection, tuning or outcome-dependent track selection. No MusicNet data or protocol changes. No GPU use.

## Immutable inputs and population

Use paper1/revision_20260915/companion/analysis_arrays.npz and tracks.json. Select exactly the previously used 133 tracks without any overlapping annotated frames, using every original note including intervals shorter than four frames when determining overlaps. Preserve original hop, boundaries, frame count and target arrays; require 34 distinct groups. Freeze file hashes and the selected IDs before computing comparisons. Do not pool additional seeds.

## Controlled predictions

For each of Mamba and S4D, compute its predicted mean separately on every annotated interval of at least one frame. These means use predictions, never target audio. On each interval render:

1. Flat: its own predicted mean at every frame.
2. Shared shape: the same mean multiplied by the existing training-fitted shared-envelope shape, normalized to discrete mean one on this interval.

Recover the archived shape by normalizing the stored shared-envelope prediction within the interval. On a nonoverlapping note this cancels the original positive Note OLS level. Verify stored Note OLS is constant and positive on each note, and independently recover the same normalized shape from shared-envelope/Note-OLS ratios. No shape fitting on evaluation targets occurs. Reject rather than silently repair a nonpositive or nonfinite template mean.

In both renderings retain the neural predictor's original values outside annotated notes. Thus activity gaps, horizon and note means match across the two controls, and changes are confined to within-note shape. These controls are not the original neural output and are not proposed deployable predictors. Report original neural metrics only as descriptive anchors, not as the primary contrast.

## Metrics and verification

Primary comparison: shared-shape minus flat Full (ordinary full-horizon Pearson), reported separately for both prespecified predictors. Track means have equal weight. Report mean differences, per-track directions and 95% paired group-bootstrap percentile intervals from 5,000 draws, RNG seed 20260918. Resample the 34 whole groups with replacement and retain all selected tracks per group in each draw. These are descriptive, unadjusted intervals for two prespecified retrospective contrasts, not familywise-confirmatory significance claims.

Verify on each note that both renderings preserve the original predicted mean (relative error <=1e-10, denominator floor1e-7). Verify Note and uncalibrated note-log-RMSE remain equal (absolute difference <=1e-10). Match the existing four-frame note eligibility rule and correlation definition (at least three samples; standard deviation >1e-10). Within correlations for constant notes are undefined and must remain absent, not zero-filled. Report coverage explicitly. Record finite arrays, original target hash, track identity, group and frame counts; do not use target statistics to choose renderings.

Replay the original Note-OLS/shared-envelope Full aggregate on the same 133 tracks as an implementation check (.467 and .546 after rounding to three decimals); retain its exact existing results. Use a per-track constant-shape identity control and an independent dot-product Pearson calculation to catch implementation errors. Save new arrays, per-track results, plan/code/input hashes and a manifest in this isolated directory.

## Interpretation and stopping

Report both prespecified predictor outcomes, favorable or not. A positive Full contrast with invariant means extends the existing controlled counterexample to stronger level sources. A null or negative contrast shows its direction depends on the level/shape pairing, and must not be relabeled successful positive replication. This experiment does not remove historical researcher selection, establish a universal effect, or decompose arbitrary neural gains. Do not add predictors or seeds in response to the result. Keep v0.44.23 and its reviews intact; any manuscript integration must be identified as a subsequent revision.
