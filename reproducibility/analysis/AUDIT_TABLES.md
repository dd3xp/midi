# Supplementary numerical results

All means weight tracks equally. Intervals are pointwise, conditional on fixed predictions. All planned numerical results, including additional pairs, are in `expected_results.json`.

## Absolute error on the complete 186-track population

|Predictor|Full linear RMSE|Active linear RMSE|Full log RMSE|Active log RMSE|
|---|---:|---:|---:|---:|
|gate|0.8595|0.9655|5.3607|3.9382|
|global_ols|0.0275|0.0297|1.0755|0.9690|
|instrument_ols|0.0264|0.0289|1.0193|0.9151|
|note_ols|0.0260|0.0280|4.0289|0.9267|
|note_shape|0.0255|0.0275|4.0196|0.8718|
|mamba|0.0240|0.0260|0.9136|0.8259|
|s4d|0.0252|0.0272|0.9357|0.8442|
|bigru|0.0223|0.0244|0.8692|0.7807|

## Population and annotation checks

Mamba minus instrument OLS, note-mean correlation. These subsets are sensitivity views of existing tracks, not newly held-out datasets.

|Population|Tracks|Groups|Mean difference|95% interval|
|---|---:|---:|---:|---|
|all186|186|44|0.0302|[-0.0047, 0.0677]|
|nonoverlap_notes_frame|186|44|0.0131|[-0.0289, 0.0528]|
|nonoverlap_notes_continuous|186|44|0.0175|[-0.0246, 0.0583]|
|no_overlap_tracks_frame|133|34|0.0225|[-0.0230, 0.0703]|
|native_single_lane95|95|27|0.0355|[-0.0080, 0.0808]|
|deduplicated163|163|44|0.0328|[-0.0007, 0.0685]|
|URMP|136|27|0.0052|[-0.0347, 0.0464]|
|non_URMP|50|17|0.0983|[0.0462, 0.1548]|

## All model views after continuous-time overlap exclusion

28,077 eligible notes remain; exclude both whole notes in every overlapping annotation pair. Full and active views retain the original horizon/mask.

|Predictor|Full|Active|Note means|Within notes|
|---|---:|---:|---:|---:|
|gate|0.4420 (n=186)|undefined|undefined|undefined|
|global_ols|0.4978 (n=186)|0.2037 (n=186)|0.1373 (n=186)|0.2990 (n=186)|
|instrument_ols|0.5104 (n=186)|0.2841 (n=186)|0.2417 (n=186)|0.3248 (n=186)|
|note_ols|0.4756 (n=186)|0.1431 (n=186)|0.1738 (n=186)|undefined|
|note_shape|0.5488 (n=186)|0.3368 (n=186)|0.1738 (n=186)|0.5227 (n=186)|
|mamba|0.6167 (n=186)|0.4473 (n=186)|0.2591 (n=186)|0.5993 (n=186)|
|s4d|0.6071 (n=186)|0.4298 (n=186)|0.2412 (n=186)|0.5856 (n=186)|
|bigru|0.5993 (n=186)|0.4258 (n=186)|0.2200 (n=186)|0.5993 (n=186)|

## Verification

An independent NumPy implementation checked 44,640 per-track metric entries, 576 paired bootstrap comparisons, overlap counts and fold-role separation. Maximum discrepancy: 3.45e-15 (tolerance 1e-8). Original primary metric reproduction maximum discrepancy: 1.78e-15. Input hashes are retained in the report. These checks establish numerical consistency, not unseen-data validity.

## Individual corpus checks

Mamba minus instrument OLS on note means. These are exploratory comparisons; Bach10 constructs offsets from the next onset, so interval conventions differ. Very small group counts provide limited uncertainty information.

|Corpus|Tracks|Groups|Difference|95% interval|
|---|---:|---:|---:|---|
|Bach10|40|10|0.0896|[0.0373, 0.1434]|
|PHENICX|2|2|0.0566|[-0.0399, 0.1532]|
|TRIOS|8|5|0.1523|[-0.0437, 0.3692]|
|URMP|136|27|0.0052|[-0.0347, 0.0464]|

## Within-note nonconstancy checks

Of 30,352 eligible target intervals, none has standard deviation below 1e-8 or 1e-6; one is below 1e-4. Minimum: 7.2617e-5; first percentile: .00075994; median: .00589944 (cached RMS units). Per-track eligible counts: min 24, Q1 57.5, median 140.5, Q3 219, max 777. Neural predictors and shared envelopes have defined correlations on all eligible notes. Note OLS: 569 notes in 53 tracks.

|Predictor|Original within-note mean|Target SD >=1e-4|Change|
|---|---:|---:|---:|
|gate|undefined|undefined|undefined|
|global_ols|0.298925|0.298931|0.0000058|
|instrument_ols|0.320473|0.320485|0.0000125|
|note_ols|0.045612|0.045612|0.0000000|
|note_shape|0.516908|0.516919|0.0000110|
|mamba|0.592058|0.592071|0.0000125|
|s4d|0.580713|0.580721|0.0000074|
|bigru|0.594726|0.594739|0.0000133|

All means are identical to the original at the two intermediate thresholds. Full per-track results and planned thresholds are supplied; these checks do not turn correlations into a calibrated perceptual accuracy scale.

## Note-mean numerical errors

Compute each note mean first, then its linear or log error, average squared/absolute errors over notes and take the root where appropriate; finally average tracks equally. The log uses epsilon 1e-7 after averaging, without fitted gain. Main-table Error is note-mean log RMSE. Full/active frame errors above remain reported separately.

|Predictor|Note linear RMSE|Note log RMSE|Note linear MAE|
|---|---:|---:|---:|
|gate|0.963498|3.708426|0.963335|
|global_ols|0.027222|0.814200|0.023670|
|instrument_ols|0.026170|0.745229|0.022023|
|note_ols|0.024914|0.723253|0.021389|
|note_shape|0.024916|0.723343|0.021391|
|mamba|0.023161|0.688952|0.019876|
|s4d|0.024761|0.718275|0.021263|
|bigru|0.021582|0.649831|0.017626|

|Paired difference|Error measure|Difference|95% interval|
|---|---|---:|---|
|mamba minus instrument_ols|note_linear_rmse|-0.003009|[-0.006230, 0.000115]|
|mamba minus instrument_ols|note_log_rmse|-0.056276|[-0.133192, 0.022958]|
|mamba minus instrument_ols|note_linear_mae|-0.002147|[-0.005260, 0.000902]|
|mamba minus note_shape|note_linear_rmse|-0.001755|[-0.004655, 0.000852]|
|mamba minus note_shape|note_log_rmse|-0.034391|[-0.106403, 0.034737]|
|mamba minus note_shape|note_linear_mae|-0.001514|[-0.004441, 0.001121]|
|mamba minus bigru|note_linear_rmse|0.001579|[-0.001146, 0.004986]|
|mamba minus bigru|note_log_rmse|0.039121|[-0.045899, 0.140723]|
|mamba minus bigru|note_linear_mae|0.002250|[-0.000509, 0.005726]|

## Alternative aggregation

Primary: arithmetic mean of per-track r. Group arithmetic: average tracks within group, then groups equally. Track Fisher: arctanh of each clipped track correlation, average tracks, then tanh. Reapply the complete estimator inside each group-bootstrap draw. These do not refit models or define a preferred estimator after observing results.

|Comparison / view|Arithmetic tracks: gap [CI]|Arithmetic groups: gap [CI]|Fisher tracks: gap [CI]|
|---|---|---|---|
|mamba minus instrument_ols / raw|0.1063 [0.0858, 0.1302]|0.1126 [0.0873, 0.1394]|0.1031 [0.0826, 0.1274]|
|mamba minus instrument_ols / active_raw|0.1632 [0.1355, 0.1943]|0.1701 [0.1388, 0.2031]|0.1645 [0.1345, 0.1974]|
|mamba minus instrument_ols / note_mean|0.0302 [-0.0047, 0.0677]|0.0343 [-0.0051, 0.0783]|0.0170 [-0.0203, 0.0583]|
|mamba minus instrument_ols / within_note|0.2716 [0.2385, 0.3078]|0.2756 [0.2402, 0.3142]|0.2764 [0.2420, 0.3160]|
|mamba minus note_shape / raw|0.0679 [0.0469, 0.0912]|0.0684 [0.0474, 0.0902]|0.0637 [0.0444, 0.0864]|
|mamba minus note_shape / active_raw|0.1105 [0.0877, 0.1320]|0.1103 [0.0871, 0.1328]|0.1126 [0.0882, 0.1358]|
|mamba minus note_shape / note_mean|0.0961 [0.0633, 0.1258]|0.0826 [0.0510, 0.1145]|0.0925 [0.0577, 0.1243]|
|mamba minus note_shape / within_note|0.0752 [0.0583, 0.0917]|0.0686 [0.0519, 0.0847]|0.0753 [0.0586, 0.0923]|
|mamba minus bigru / raw|0.0174 [0.0024, 0.0335]|0.0209 [0.0079, 0.0355]|0.0146 [0.0000, 0.0302]|
|mamba minus bigru / active_raw|0.0215 [0.0025, 0.0402]|0.0267 [0.0088, 0.0448]|0.0213 [0.0009, 0.0416]|
|mamba minus bigru / note_mean|0.0438 [0.0060, 0.0808]|0.0551 [0.0196, 0.0896]|0.0401 [-0.0010, 0.0806]|
|mamba minus bigru / within_note|-0.0027 [-0.0195, 0.0104]|-0.0069 [-0.0177, 0.0034]|-0.0030 [-0.0188, 0.0098]|

The Mamba/instrument-OLS and Mamba/shared-envelope interval signs are unchanged across all four views. The Mamba/BiGRU note-mean interval becomes cross-zero under Fisher aggregation (-.0010 to .0806), while positive under arithmetic-track/group aggregation. The Fisher full-track lower endpoint is only 0.00000477, not evidence for a stable architecture ranking.
