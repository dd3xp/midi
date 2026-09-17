# Follow-up transfer-shape check

Specified on 2026-09-16 after round-four review of v0.38 and after all MusicNet results. This is a retrospective diagnostic, not an addition to the pre-outcome confirmation plan.

Use all 21 saved MusicNet recordings and every saved predictor, with unchanged predictions/targets. Count eligible and defined Within intervals. Define frame-nonoverlap by counting every annotated interval, including notes shorter than four frames, at the original cached hop. Recompute Note/Within/Error only on original eligible intervals sharing no frame with another note. Do not concatenate masked frames into a new Full measure. If any complete recording has no shared annotated frames, compare Note OLS and shared-template Full on its original complete horizon; otherwise report no such recordings. Report all coverage and empty/undefined cases. No fitting, track trimming, selection or causal attribution from this subset.
