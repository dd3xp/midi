# Fixed-prediction analysis companion

This package accompanies the MIDI-to-RMS diagnostic manuscript. It contains 186 cached targets, aligned note annotations and MIDI features, predictions for eight internal predictors, four folds, analysis code, and provenance. It reproduces analysis of the frozen predictions; it does not retrain models or recover the original recordings.

Use Python and NumPy (verified with NumPy 2.4.3); the additional public-RMS verifier also uses SciPy. No GPU, network connection, external dataset path, or API key is needed. From this directory:

```text
python check_manifest.py
python verify.py
python variance_audit.py
python level_and_aggregation_audit.py
python verify_level_audit.py
python verify_public_rms.py
```

`verify.py` independently recomputes 44,640 metric entries and 576 paired-bootstrap comparisons, and checks group separation within folds. Its implementation differs from the initial audit. The variance and aggregation audit programs reproduce the later analyses; running copies of those scripts is a portability check. Separately, verify_level_audit.py independently checks 4,464 note-error entries and 45 paired intervals through cumulative sums and explicit row resampling.

- `methods.pdf`: precise methods and readable supplementary tables.
- `METHODS.md` and `AUDIT_TABLES.md`: editable sources of that PDF.
- `analysis_arrays.npz`, `tracks.json`, `folds.json`: arrays and their identities/roles.
- `expected_results.json`: complete initial diagnostic audit, including all defined populations and negative or positive exceptions.
- `variance_audit.json` and `level_and_aggregation_audit.json`: later sensitivity results.
- `error_metrics_explicit.json`: mapping of legacy full/active-frame error fields and note-mean errors. Main-table Error is `note_log_rmse`.
- `source/`: archived preprocessing, training, model and protocol records. Their presence does not supply historical dependencies or checkpoints.
- `manifest.json`: hashes of every supplied file except the manifest itself.

The public-model protocols, reports, and 232 archived final RMS arrays are supplied. `verify_public_rms.py` checks 1,392 per-track values, 36 subset means and 120 primary paired intervals from these arrays. It does not generate audio or re-extract RMS from waveforms. `public_rms_manifest.json` records mapping and hashes of the original RMS/completion files; `public_expected_results.json` retains the unchanged reports. Public weights were not retrained on the internal folds and may have seen evaluation pieces.

This is a local working/submission companion. It has not been uploaded to a public repository. Original audio, training checkpoints, and a complete historical software environment are absent. Historical test-driven development is not removed by the current audits. See the manuscript and methods for the precise limits of inference.

## Refinement analyses

The current readable supplement is `analysis_supplement.pdf`; the older `methods.pdf` is retained for its additional audit tables. New programs and frozen results live in `refinement/`.

```text
python refinement/verify_refinement.py
python refinement/verify_scale.py
python refinement/verify_all_predictors.py
```

These programs independently check 186-track direction/interval results and full/active scale results. The latter uses explicit Gaussian kernels and FFT convolution rather than the original ndimage filter. The plan documents identify when each check was specified relative to prior results. Recomputing plots additionally needs Matplotlib; running the independent numerical verifiers needs NumPy and SciPy.

`python refinement/log_error_parts.py` recomputes the overall log offset and centered note-error components for all eight predictors and checks the exact log-MSE identity and original RMSE values. It writes a new result timestamp; verify the shipped manifest before running programs that regenerate included reports.

`python refinement/duration_check.py` reproduces all four minimum-note-length conditions, defined coverage and six paired Within comparisons, checking the primary condition against archived values. It also writes a new timestamp.

## Current corpus and public-output diagnostics

`acceptance/corpus_analysis.py` and `acceptance/public_directions.py` reproduce the additional paired corpus comparisons and public-pipeline direction counts. All primary populations are retained. `PLAN_corpus_frozen.md` is the exact plan at the corpus computation; the later `PLAN.md` additionally specifies the public-output extension. The programs write their results, so verify the manifest first and work in a copy.

The separate `paper1_execution_companion_20260916.zip` supplies the corrected training cache, actual model/training source, all 12 retained seed-42 checkpoints, and commands for CPU replay or a fresh single-GPU fold. Raw recordings are not contained in either package. These files support the cached-input experiment and analysis, not a new claim of exact raw-audio target reconstruction.
