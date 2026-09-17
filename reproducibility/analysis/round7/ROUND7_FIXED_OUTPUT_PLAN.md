# Post-review checks on fixed outputs (round 7)

These are retrospective diagnostics, not a new prospective validation. Do not change predictions, targets, folds, training, or model selection.

1. Recompute Note/Error support for all retained predictors and all 186 tracks, using original integer truncation, four-frame intervals and correlation thresholds. Report exceptions rather than dropping them silently.
2. Recompute mean squared log offset and centered variance for Mamba, shared envelope, instrument OLS and BiGRU, verifying existing supplement values.
3. Audit file-end padding sensitivity without reconstructing missing recordings: exclude the first and last two RMS frames from Full; exclude every note interval touching those margins from Note. Apply this rule uniformly to all tracks and compare Mamba/instrument OLS and shared envelope/Note OLS. For centered RMS with a 320-sample window and 160-sample hop, zero versus reflected file-end padding can affect endpoint windows; this guard checks reliance on those frames, not unknown preprocessing in general or annotation offsets. Do not present this as re-extraction with a known original configuration.
4. Preserve the already chosen median/IQR discordant example and its central ten-second window from Supplement B. Show all full-track eligible note means and all 186 track differences. Do not select a more striking example or a locally favorable time window.

No new outcome is used to choose a cutoff or sample. Case eligibility and window are inherited from the archived illustrative analysis; their selection was after initial aggregate inspection and is not preregistration.
