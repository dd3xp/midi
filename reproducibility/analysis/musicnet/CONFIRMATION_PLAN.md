# Frozen MusicNet transfer confirmation, 2026-09-16

This plan is fixed before MusicNet model inference or outcome inspection. It adds a new-recording transfer test to the retrospective 186-track diagnostic study. It does not erase development on the original evaluation data. The user explicitly states that MusicNet has never been used in the earlier ISMIR, AAAI or related development. Their newly supplied manuscripts describe the existing four corpora, not MusicNet.

## Population and identity

Select every row of the official release metadata whose ensemble is Solo Violin or Solo Cello: 21 recordings/movements, nine violin and twelve cello, in five catalog works (BWV 1001, 1002, 1006, 1009, 1010). Selection uses metadata only. Retain all these recordings regardless of scores, duration, polyphony or recording level. Solo instrumentation does not imply monophonic playing. Audit work identity against the earlier repertoire before calling this an unseen-work test; any unresolved identity is disclosed, never silently excluded. Do not treat 21 movements as 21 independent compositions.

Use the official MusicNet raw audio plus audio-aligned CSV labels, release archive MD5 844764911fa0d5b97c97da944a057590. Original reference MIDI timing is not aligned and will not supply performed onset times. Preserve each WAV and CSV hash and the original metadata/source field. Verify the release checksum and exactly one WAV/CSV pair per selected ID. No raw recording redistribution.

## Frozen predictors and roles

Use the original seed-42 fold-0 Mamba and S4D best checkpoints, chosen as the first numbered fold before observing new outcomes. Do not average checkpoints, retrain, fine-tune or choose a fold/seed by MusicNet results. Use the saved fold-0 global OLS, instrument OLS, Note OLS and shared-envelope coefficients/templates, fitted on the same original training role. Include the timing gate as a diagnostic reference. This is a fixed transfer setting, not an outer-fold MusicNet training experiment. Record each checkpoint, coefficient, data, split and source hash.

Primary paired comparisons: Mamba minus instrument OLS and shared envelope; S4D with the same two references; shared envelope minus Note OLS as the level-preserving shape control. Report all candidate results even if no disagreement or no neural gain occurs. Polyphonic note intervals are intervals of the entire solo recording, not isolated-note energies. Preserve overlapping intervals; the frozen 20-feature extractor's onset order and overwrite behavior remain unchanged and are audited.

Decision check: evaluate the five fixed candidates {instrument OLS, Note OLS, shared envelope, Mamba, S4D} on all original fold-0 validation tracks, using full sequences. Select one candidate by equal-track mean Full and one by equal-track mean Note; only candidates with both metrics defined on every validation track are eligible. Ties within 1e-12 use lexicographic model name. Save this selection before MusicNet inference. The checkpoints were originally chosen by their fixed-window validation loss; this secondary model-choice illustration does not change them. If both rules choose the same predictor, report that outcome rather than searching for a reversal. Global OLS and the gate are descriptive controls, not additional selection candidates.

## Input and target conversion

Read original PCM WAV values as float32 with no peak, track or corpus gain normalization. Average channels if non-mono, recording this action. Resample to 16,000 Hz using librosa 0.11.0 soxr_hq. RMS: frame_length=320, hop_length=160, center=True, pad_mode='constant', float32. Keep the complete recording horizon, including initial silence and final tail. No trimming or score-dependent interval selection. This explicit new-target pipeline is not claimed to recover unknown historical cache padding.

CSV start_time and end_time are original audio-sample indices: divide by the actual original WAV sample rate. Retain original onsets and offsets, pitches and CSV row ordering; no duration accumulation, next-onset substitution, quantization to score time or overlap removal. Convert the note array to float32, velocity zero. Use the frozen notes_to_frame_features with n_features=20, instrument vn/vc and hop_time=float(np.float32(0.01)), matching the cached feature convention. Audit source-time to float32 errors, out-of-horizon notes, simultaneous starts and overlap counts. Features use annotations and instrument identity only; no waveform statistic is an input.

## Inference and resource constraints

Unchanged architecture/state dictionaries, float32, eval mode, full recording at once, inference batch one, deterministic seed 42. GPU allocator limit 0.80, one inference job per device and process/GPU preflight before launch. Local RTX 5070 and Squishy RTX 3070 Ti are authorized. Save exact commands, environment, source/data hashes, logs, status and completion criteria before persistent hidden launch. First check full-length feasibility with original training features repeated to 34,000 frames, without new target outcomes. If a runtime-only attention backend change is necessary, document it and verify short-sequence numerical agreement before any new predictions; do not change attention context, truncate recordings or silently reduce precision. An OOM/killed GPU task is paused and is not automatically restarted or migrated.

## Outcomes and uncertainty

Use the existing analysis definitions: Full, Active, Note, Within, note log-RMSE Error, and E^2=b^2+c^2 with epsilon=1e-7. Note minimum four frames, correlation minimum three samples and standard deviation greater than 1e-10. Preserve undefined metrics and their denominators. Verify finite nonnegative predictions, identical target/prediction length, unique IDs and full 21-recording coverage. Save predictions and per-recording metrics, paired differences and Full-up/Note-down counts. Analyze the same-output linear/log/smoothing views using the original fixed scales (0, .05, .1, .2, .5, 1 second), without using them to select a predictor.

Report equal-recording and equal-work summaries and the five work-specific paired effects. An exploratory 5,000-resample composition bootstrap (seed 20260916, percentile 95%) resamples the five whole works and retains all their recordings. With only five independent works, intervals are descriptive and do not establish broad ten-instrument population generalization. Do not pool these recordings with the old 186 or pool repeated predictions as independent observations. Primary success is a valid fixed-protocol external diagnostic, not a predetermined effect direction. Report non-replication, domain shifts and selection-rule agreement as results.

## Change control

Hash this plan and all frozen inputs before original-validation predictions or MusicNet model outputs. Later bug fixes require a timestamped explanation, exact change and verification; outcome-dependent protocol changes are exploratory and separate. No further dataset/model/seed search to obtain the desired sign. The approved main figure is preserved. Only a substantively changed, verified manuscript proceeds to a new three-reviewer PDF-only round.
