# Sources and scope

The included arrays are derived features, annotation intervals, RMS targets, and model outputs. Raw recordings are not included. Track identities, connected-work groups, and the fixed fold roles are preserved in `analysis/tracks.json`, `analysis/folds.json`, and the MusicNet companion. Original dataset terms continue to apply to derived data; this release does not assign a blanket new license to third-party material.

The retrospective study uses URMP, Bach10, PHENICX-Anechoic, and TRIOS. Dataset citations and the precise construction of the 186-track population are in `analysis/methods.pdf` and `analysis/METHODS.md`. Retained preprocessing and historical source snapshots are supplied in the execution archive. Its cached features and targets support checkpoint replay; they are not substitutes for the original audio in an audio-preprocessing replication.

The independent study uses 21 previously unused MusicNet recordings representing five works. MusicNet is by John Thickstun, Zaid Harchaoui, and Sham M. Kakade: [official release, DOI 10.5281/zenodo.5120004](https://zenodo.org/records/5120004). The release contains recording-specific provenance and Creative Commons/public-domain source information. Consult it for the applicable recording terms. This repository retains acquisition hashes and the selected-track identity audit under `analysis/musicnet/provenance/`.

MIDI-DDSP and MIDI2Params remain separately identified adapted public systems. Their metric arrays permit checking the reported final-audio RMS comparisons. The release does not claim that these arrays reproduce public-system training or synthesis from scratch; see `analysis/PROTOCOL.md` and `analysis/public_rms_manifest.json` for the stored system provenance.

Historical research plans and source snapshots retain their original wording and paths. Statements there about a local-only bundle describe the archived research stage. The current public entry point and runnable commands are the parent README, not those historical paths.
