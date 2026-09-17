# Release validation

Checked on 2026-09-17 against the v0.44.1 manuscript's retained outputs.

| Check | Result |
|---|---|
| Clean Python 3.12.9 environment, NumPy 2.4.3, SciPy 1.17.1 | All 16 fixed-output analysis checks passed |
| Core independent verifier | 44,640 metric entries and 576 paired-bootstrap comparisons checked |
| Retrospective population | 186 unique tracks, four fixed folds, 44 connected groups |
| Frozen MusicNet transfer | 21 recordings, five works; complete fixed-output verification passed |
| Checkpoint execution | All 12 seed-42 checkpoints loaded strictly; CPU replay of one full track per fold/model agreed at rtol=1e-3, atol=1e-5 |
| Frozen MusicNet checkpoint execution | Both fold-0 models replayed the shortest complete recording (10,285 frames); maximum absolute difference below 7.5e-8 |
| Additional seeds | Mamba and S4D seeds 123/456, all four folds each; 186 unique tracks per seed; per-seed metrics and paired analysis reproduced |
| Seed result comparison | 4,089 floating-point entries per model matched the archived seed-results JSON exactly on this machine |
| Archive integrity | SHA256 and file sizes are pinned in assets.json; internal archive manifests also checked |

The release runner caps CPU numerical-library threads and never starts training. Four historical audit scripts used exact equality for nested floating-point JSON results. BLAS thread settings changed their final floating-point bits (observed maximum 3.53e-15 in the transfer-shape audit). The portable copies now compare floating-point values with rtol=atol=1e-12 while keeping structure, counts, identities and hashes exact. Stored arrays, expected results, plans and manuscript numbers were not changed. The analysis manifest includes the updated script hashes.

Checkpoint replay here is an execution check on twelve complete tracks, not a fresh 186-track inference run. The full-coverage metric check uses every stored prediction. Use replay_checkpoints.py --all-tracks for complete neural inference. Training source is provided, but this release validation did not retrain the models. Archived research notes may describe earlier local-only package states; the parent README specifies the public supported commands.

Machine-readable verification records are in validation/. They report runtime checks, not new experimental results or cross-hardware efficiency comparisons.

The public checkpoint and MusicNet replay entry points write new reports outside the downloaded execution archive. This avoids the historical replay script overwriting its own manifest-covered result file. Checkpoint contents, model code and predictions remain unchanged.
