# Shape control at fixed neural-predicted note levels

This additional retrospective analysis uses the existing seed42 out-of-fold Mamba and S4D predictions. It fixes each model's predicted note means and all predictions outside annotated intervals, then compares constant note shapes with the original training-fitted shared template. No training, new model selection or MusicNet fitting occurs.

The fixed population is 133 nonoverlapping tracks from 34 groups. Equal-track Full gains for shared versus constant shape are .069 [.052, .089] and .070 [.052, .090], respectively; Note and uncalibrated Error remain unchanged. These compare two controlled renderings, not performance improvements over the original neural outputs. See `results.json` for original-predictor anchors, all directions and coverage, and `PLAN.md` for the plan saved before the new comparisons.

## Reproduce

Use the repository's pinned numerical environment. From `reproducibility/`:

```sh
python analysis/neural_level_control/reproduce.py --inputs analysis
```

For the local paper delivery bundle, from the bundle root:

```sh
python strong_level_control/reproduce.py --inputs analysis_companion
```

No GPU, new audio, external service or checkpoint download is needed. The runner copies the three existing input files and their manifest to a fresh directory under `runs/`, executes the original analysis and an independent verifier, and compares every per-track and summary result to the retained output at absolute tolerance 1e-12. It refuses to overwrite an existing output directory. An optional `--output PATH` sets a new destination. The approximately 57MB source cache is copied, not modified.

`prepare.py` verifies original hashes, four-fold role separation and unique track coverage before freezing the selected IDs. `analyze.py` checks mean conservation, unchanged gaps, Note/Error invariance and undefined constant-shape Within values, then uses 5,000 whole-group bootstrap draws with seed20260918. `verify.py` checks saved arrays by independent interval sums, dot-product Pearson calculations and explicit group concatenation. The original Note-OLS/shared-template result is reproduced as a control.

The source scripts retain their historical relative directory layout; the runner supplies that layout in its fresh directory. The result is an additional check on a previously studied population. It neither removes researcher-selection history nor establishes general effects across corpora. Both prespecified predictor contrasts are reported; confidence intervals are descriptive and unadjusted, not familywise-confirmatory tests.
