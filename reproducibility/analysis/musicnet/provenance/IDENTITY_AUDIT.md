# Repertoire identity audit

MusicNet selection is metadata-only: BWV 1001/1002/1006 (solo violin), BWV 1009/1010 (solo cello), 21 movements in five catalog works. The user states no previous MusicNet use. The two supplied earlier manuscripts list only the original four corpora. This supports independent recording exposure; independence is not inferred merely from a new dataset name.

URMP's final official documentation has abbreviated names, including an ambiguous Sonata. We checked the authors' original paper, arXiv:1612.08727v1, Table II (downloaded as urmp_author_v1.pdf). It identifies Sonata as Mozart K.331, Nocturne as Mendelssohn, Arioso as Bach's cantata BWV 156, and Chorale as Saint-Saens. Fugue is The Art of the Fugue. None is a selected solo sonata, partita or cello suite. The final 2018 URMP documentation preserves the same named repertoire, although some durations differ from the early paper; we do not use its old durations for time conversion. The author-hosted later paper is preserved separately as urmp_author_paper.pdf.

Primary sources:
- https://arxiv.org/pdf/1612.08727v1 (Table II, repertoire names)
- https://labsites.rochester.edu/air/projects/URMP/URMP_doc.pdf (final dataset names and file definitions)
- https://zenodo.org/records/5120004 (official MusicNet release and metadata)
- https://raw.githubusercontent.com/jthickstun/pytorch_musicnet/master/musicnet.py (author's loader: CSV start/end indices index the original sample array; do not use start_beat/end_beat as seconds)

The existing Bach10 entries are the ten named chorales; PHENICX and TRIOS use different Beethoven/Mozart/Brahms/Schubert/other ensemble repertoire. Original cache identities and MusicNet catalog IDs are retained in the confirmation manifest. This is a work-identity check, not a claim of independent style, composer or instrumentation: the new set is entirely Bach and strings. Recording-source and annotation differences remain part of transfer. The full data-adapter audit will verify the actual label programs, waveform horizons, overlaps and file hashes before inference.
