# Acceptance-oriented refinement, 2026-09-16

Authorization: user requested continued fresh-agent review/refinement and early supplementary experiments where useful. One reviewer at a time, no previous context. The first reviewer receives only the current main PDF and the question whether it could be accepted at ICASSP. Reviewer agreement is not a probability of acceptance.

Parent retains the ISMIR issue map and anti-defensive writing guidance, while preserving unfavorable results, historical selection disclosures, and distinctions between correlation and numerical agreement.

## Frozen additional seed study (before any new predictions)

- Existing seed 42 outputs remain untouched. Add seeds 123 and 456 to Mamba and S4D on all four existing outer folds: 16 new jobs.
- Copy the completed personal-device training program and source snapshot byte-for-byte. Change only the model seed and descriptive execution metadata in each isolated protocol; retain model, features, crops, batch 32, losses, optimizer, validation windows, stopping rule, and folds.
- Use the existing corrected training cache, including its original pitch auxiliary targets. Verify features, RMS targets, notes and fold membership against the paper's analysis companion before launch. Do not reconstruct training recordings or f0 from RMS.
- Allocate Mamba to the local 5070 and S4D to Squishy 3070 Ti, one training process per machine, allocator cap 80%. Existing seed-42 runs include earlier runtime migrations; the new repetitions characterize robustness under the retained recipe, not exact historical floating-point replay.
- Save source/data hashes, runtime versions, commands, logs, atomic checkpoints and completion markers. No automatic retries, no migration after kill/OOM, no change to stopped historical schedulers. A new bounded queue ends after its eight assigned jobs or its first failed job. It is not a scheduler renewal.
- Report each seed separately: Full/Active/Note/Within/Error, paired model-reference changes, Full-up/Note-down and Full-up/Error-up counts and magnitudes against instrument OLS and the shared envelope. No best-seed choice, no ensemble, no treating track-by-seed copies as independent tracks. Report incomplete coverage if not all jobs finish.

## Corpus check (before computation)

Use frozen seed-42 arrays for all three neural predictors and both references, separately for all four corpora and the non-URMP aggregate. Record tracks/groups and the same direction counts and magnitudes. Keep small-corpus results descriptive; do not infer robustness from PHENICX's two tracks. Recompute per-track quantities from arrays and reconcile full-population counts with the audited paper table.

## Reproducibility and submission

Inventory the supplied AAAI zip without executing its training/preprocessing scripts. Compare source and metadata to the ICASSP snapshot; retain different splits/teacher protocols as historical, never silently substitute them. Prepare a clean current-paper supplement/code package without internal reviews or editorial history. Check official 2027 rules; do not invent an upload slot or claim a local package is publicly accessible. Main paper must support its conclusions without optional material. No public upload or submission is authorized in this turn.
