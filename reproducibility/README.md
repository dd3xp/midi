# Controlled evaluation of MIDI-to-RMS prediction

Reproducibility materials for **Disentangling Note Levels and Envelope Shape: A Controlled Evaluation of MIDI-to-RMS Prediction**.

This directory is the entry point for the current study. The repository's older experiments remain historical work and use different configurations. No conference acceptance is implied by the directory or release names.

## 1. Recompute the paper's fixed-output analyses

Use Python 3.12. The numerical environment is pinned in `requirements.txt`.

```sh
cd reproducibility
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python reproduce.py
```

No GPU, raw recordings, API key, or external dataset directory is needed. The analysis arrays are included in the repository. The runner copies immutable input material into a new `runs/` directory and writes logs plus `report.json` there. Repeated runs do not overwrite the shipped arrays or expected results. `python reproduce.py --quick` runs the manifest and core metric/paired-bootstrap checks only.

The full command checks:

- all 186 retrospective tracks and four group-disjoint folds;
- Full, Active, Note, Within and note-log-RMSE results and paired uncertainty;
- public-synthesizer final RMS arrays, amplitude-domain and smoothing comparisons;
- the mean-preserving shape intervention, overlap, duration and endpoint checks;
- the frozen MusicNet transfer: 21 recordings belonging to five works;
- corpus-specific and track-level comparison directions.

Each checker asserts agreement with the retained reference outputs. The basic independent verifier checks 44,640 metric entries and 576 paired-bootstrap comparisons. See `analysis/METHODS.md`, `analysis/PROTOCOL.md` and the per-analysis directories for definitions and historical frozen plans. Archived plans document when choices were made; they are not external preregistrations.

## 2. Replay checkpoints and reproduce additional seeds

Larger cached inputs and checkpoints are distributed as SHA256-verified GitHub Release assets rather than Git objects. Install PyTorch 2.10.0 in the same environment (for CPU replay: `python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu`), then run:

```sh
python fetch_assets.py execution
python replay_checkpoints.py
```

This performs CPU inference using all twelve seed-42 checkpoints, one complete evaluation track per checkpoint, compared with stored predictions at `rtol=1e-3, atol=1e-5`. Add `--all-tracks` to the replay command for complete inference. Results go into a new file under `runs/`; the downloaded archive remains unchanged. The default replay is an execution check, not a substitute for the full-coverage metric reproduction above. PyTorch 2.10.0+cu128, NumPy 2.4.3 and SciPy 1.17.1 were used for the local execution check; no cross-hardware bitwise identity is claimed.

```sh
python fetch_assets.py mamba-seeds s4d-seeds
python analysis/acceptance/analyze_seeds.py --model mamba --root assets/mamba-seeds --execution assets/execution --output runs/mamba_seeds.json
python analysis/acceptance/analyze_seeds.py --model s4d --root assets/s4d-seeds --execution assets/execution --output runs/s4d_seeds.json
```

Run the fixed-output command first so `runs/` exists. The seed programs verify checkpoint/prediction hashes, fold roles, timeline correspondence, finite outputs and 186 unique tracks per seed, then recompute metrics and paired comparisons separately. They do not pool the repeated tracks into 558 independent observations. The last Mamba seed456/fold3 job ran on a different device, as recorded in its provenance.

Frozen MusicNet neural predictions can also be replayed from the same execution asset:

```sh
python replay_musicnet.py
# Optional: replay both models on all 21 complete recordings
python replay_musicnet.py --all-tracks
```

The default checks each model on the shortest complete recording. It verifies the pre-outcome fold-0 checkpoint hashes, uses the retained features without fitting, and compares CPU predictions with the archived transfer at `rtol=1e-4, atol=1e-6`. Full-recording attention can require substantial CPU memory. This does not change the transfer model choice or protocol.

Use these root-level replay entry points. The execution archive preserves the historical `replay.py`, which writes its original result path and is not the repeatable public entry point.

Fresh training commands are documented in `assets/execution/README.md`. They require a new output directory and a compatible GPU, and do not overwrite checkpoints. The training source and original protocol are retained, but restarting on different hardware need not reproduce an identical floating-point training trajectory. Running this repository's default commands does not launch GPU training.

## Scope and provenance

The 186-track study is retrospective: model development had access to this population. Additional seeds do not remove that history. MusicNet recordings were previously unused and the transfer choices were frozen before their outputs. Results retain unfavorable comparisons and the absence of average template gain on MusicNet.

The package supplies derived targets, features, annotations, predictions, frozen splits, source and checkpoints. Original recordings and the public synthesizers' audio/weights are not bundled. The public-system checker reproduces metrics from final RMS arrays; it does not rerun audio synthesis. The historical RMS padding/library setting is not fully recoverable, so the cache is the reproducible target rather than a claim of exact reconstruction from original audio.

`analysis/source/` and `analysis/musicnet/source/` preserve historical source snapshots. The supported portable entry points are the commands above; original acquisition scripts may retain their execution-site paths. Source datasets and third-party components retain their respective attribution and terms. See `DATA_SOURCES.md` before redistributing data or third-party components.

Publication/release validation results are recorded in `VALIDATION.md`. The manuscript is not automatically submitted or published by these scripts.
