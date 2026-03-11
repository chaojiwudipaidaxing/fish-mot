# Sanity Check (section-level traceability)

RESULTS_ROOT = `results/archive/main_val_20260302_TESTCOMMIT1234567_final2`

## 0) Abstract numbers traceability

| Number used in Abstract | Value | Source and extraction rule |
|---|---:|---|
| Sequence count in frozen snapshot | 8 | `tables/per_seq_main_val.csv`: unique `seq_name` count |
| Frame budget per sequence | 1000 | `run_config.json`: `max_frames` |
| Base HOTA mean/std | 58.232 / 0.304 | `tables/main_table_val_with_baselines.csv`, row `method=Base`, cols `HOTA_mean`,`HOTA_std` |
| Base IDF1 mean/std | 64.042 / 0.550 | `tables/main_table_val_with_baselines.csv`, row `method=Base`, cols `IDF1_mean`,`IDF1_std` |
| ByteTrack HOTA mean/std | 58.431 / 0.291 | `tables/main_table_val_with_baselines.csv`, row `method=ByteTrack`, cols `HOTA_mean`,`HOTA_std` |
| ByteTrack IDF1 mean/std | 64.011 / 0.545 | `tables/main_table_val_with_baselines.csv`, row `method=ByteTrack`, cols `IDF1_mean`,`IDF1_std` |

## 1) Main and combined results (source: `tables/main_table_val_with_baselines.csv`)

| Key value used in main.tex | Value | Source (row,col) |
|---|---:|---|
| Base HOTA_mean | 58.232 | row=0, col=HOTA_mean |
| Base IDF1_mean | 64.042 | row=0, col=IDF1_mean |
| ByteTrack HOTA_mean | 58.431 | row=4, col=HOTA_mean |
| ByteTrack-IDSW vs Base-IDSW gap | 9.000 | rows=4,0 col=IDSW_mean |
| +gating HOTA_mean | 50.345 | row=1, col=HOTA_mean |

## 2) Strong baselines (source: `tables/strong_baselines_seedmean.csv` + `tables/strong_baselines_seedstd.csv`)

| Key value used in main.tex | Value | Source (row,col) |
|---|---:|---|
| ByteTrack HOTA mean | 58.431 | strong_mean row=0, col=HOTA |
| ByteTrack HOTA std | 0.291 | strong_std row=0, col=HOTA |
| OC-SORT IDF1 mean | 52.631 | strong_mean row=1, col=IDF1 |
| BoT-SORT IDSW mean | 121.000 | strong_mean row=2, col=IDSW |
| ByteTrack-Baseline HOTA gap | 0.199 | with_baselines rows 4 and 0, col=HOTA_mean |

## 3) Controlled degradation (source: `tables/degradation_grid.csv`)

| Key value used in main.tex | Value | Source (row,col) |
|---|---:|---|
| Base Delta HOTA @j=0.00 | 35.707 | rows 1 and 13, col=HOTA |
| Base Delta IDF1 @j=0.00 | 27.959 | rows 1 and 13, col=IDF1 |
| +gating Delta HOTA @j=0.02 | 24.139 | rows 3 and 15, col=HOTA |
| ByteTrack Delta IDF1 @j=0.02 | 25.082 | rows 5 and 17, col=IDF1 |
| Worst-point top HOTA (ByteTrack) | 41.435 | row=17, col=HOTA |
| Worst-point IDSW order anchor (Base) | 99.625 | row=16, col=IDSW |

## 3.1) Gating-threshold sensitivity (source: `tables/gating_sensitivity.csv`)

| Key value used in main.tex | Value | Source (row,col) |
|---|---:|---|
| Base HOTA | 57.831 | row=0, col=HOTA |
| +gating HOTA @1000 | 51.469 | row=1, col=HOTA |
| +gating HOTA @2000 | 51.469 | row=2, col=HOTA |
| +gating HOTA @4000 | 51.469 | row=3, col=HOTA |
| +gating IDF1 plateau | 55.202 | rows=1/2/3, col=IDF1 |
| Base to +gating IDSW gap | 51.875 | rows=1 and 0, col=IDSW |
| Base to +gating HOTA gap | -6.362 | rows=1 and 0, col=HOTA (`+gating - Base`) |
| Base to +gating IDF1 gap | -8.072 | rows=1 and 0, col=IDF1 (`+gating - Base`) |

## 4) Stratified failure modes (source: `stratified/stratified_metrics_val.csv`)

| Key value used in main.tex | Value | Source (row,col) |
|---|---:|---|
| Base turning (mid) IDSW | 627 | row=28, col=IDSW |
| Base low-confidence (mid) IDSW | 612 | row=40, col=IDSW |
| Base occlusion-low CountMAE | 4.655 | row=0, col=CountMAE |
| +gating low-confidence (mid) IDSW | 984 | row=41, col=IDSW |
| +gating occlusion-high CountMAE | 1.848 | row=9, col=CountMAE |

## 5) Counting stability (source: `tables/count_metrics_val.csv`, `row_type=mean`)

| Key value used in main.tex | Value | Source (row,col) |
|---|---:|---|
| Base CountMAE | 2.321 | row=8, col=CountMAE |
| +gating CountMAE | 2.379 | row=17, col=CountMAE |
| +traj CountRMSE | 2.911 | row=26, col=CountRMSE |
| +adaptive CountDrift | 2.100 | row=35, col=CountDrift |
| Base CountVar | 3.746 | row=8, col=CountVar |

## 6) Runtime and deployment (source: `runtime/runtime_profile.csv`)

| Key value used in main.tex | Value | Source (row,col) |
|---|---:|---|
| Base FPS | 822.812 | row=0, col=fps |
| +gating mem_peak_mb | 548.395 | row=1, col=mem_peak_mb |
| +traj cpu_mean_norm_percent | 69.259 | row=2, col=cpu_mean_norm_percent |
| +adaptive FPS | 89.024 | row=3, col=fps |
| Base total_frames | 8000 | row=0, col=total_frames |

## 6.1) Discussion deployment-threshold numbers

| Number used in Discussion | Value | Source (row,col) |
|---|---:|---|
| Deployment FPS threshold | 30 | manuscript policy threshold (not measured metric) |
| Memory threshold | 600 MB | manuscript policy threshold (not measured metric) |
| Base FPS shown as 822.8 | 822.812 | `runtime/runtime_profile.csv`, row=0, col=fps (rounded to 1 decimal) |
| +gating FPS shown as 1182.2 | 1182.212 | `runtime/runtime_profile.csv`, row=1, col=fps (rounded to 1 decimal) |
| Base mem shown as 549.0 MB | 548.992 | `runtime/runtime_profile.csv`, row=0, col=mem_peak_mb (rounded to 1 decimal) |
| +gating mem shown as 548.4 MB | 548.395 | `runtime/runtime_profile.csv`, row=1, col=mem_peak_mb (rounded to 1 decimal) |

## 7) Reproducibility statement (source: `release/manifest.json`)

- git_commit: `TESTCOMMIT1234567`
- input hash count: 10
- output hash count: 9

## 8) Path integrity self-check (`main.tex`)
- Check 1: no `results/main_val` or `results\main_val` literal.
- Check 2: no non-final2 archive root.
- Result: PASS

## 9) Figure/Table -> source mapping (frozen bundle)

| Manuscript object | Paper file used | Frozen source for audit |
|---|---|---|
| Core methods table | `paper_assets/paper_main_table.tex` | `tables/main_table_val_seedmean.csv` + `tables/main_table_val_seedstd.csv` |
| Core methods figure | `paper_assets/main_table_val_seedmean_std_paper.png` | same source as core table |
| Strong baselines table | `paper_assets/strong_baselines_seedmean_std.tex` | `tables/strong_baselines_seedmean.csv` + `tables/strong_baselines_seedstd.csv` |
| Strong baselines figure | `paper_assets/strong_baselines_seedmean_std.png` | same source as strong table |
| Combined methods table | `paper_assets/main_table_with_baselines.tex` | `tables/main_table_val_with_baselines.csv` |
| Combined methods figure | `paper_assets/main_table_with_baselines.png` | `tables/main_table_val_with_baselines.csv` |
| Degradation figure | `paper_assets/degradation_grid.png` | `tables/degradation_grid.csv` |
| Degradation delta table | inline in `main.tex` | computed from `tables/degradation_grid.csv` (drop 0.0 -> 0.4) |
| Gating sensitivity figure | `paper_assets/gating_sensitivity.png` | `tables/gating_sensitivity.csv` |
| Gating sensitivity table | `paper_assets/gating_sensitivity.tex` | `tables/gating_sensitivity.csv` |
| Stratified figure | `paper_assets/stratified_metrics_val.png` | `stratified/stratified_metrics_val.csv` |
| Count stability table | `paper_assets/paper_count_table.tex` | `tables/count_metrics_val.csv` |
| Count stability figure | `paper_assets/count_stability_bar_paper.png` | `tables/count_metrics_val.csv` |
| Runtime figure | `paper_assets/runtime_profile.png` | `runtime/runtime_profile.csv` |

## 10) Additional prose-number coverage (Results/Discussion)

| Numbers in manuscript prose | Source and rule |
|---|---|
| 28.299, 19.257, 35.557, 27.857, 33.474, 25.141, 14.888, 33.382 | `tables/degradation_grid.csv`, Delta rule: value(drop=0.0) - value(drop=0.4), grouped by method and jitter |
| 41.182, 38.727, 49.770, 45.989, 112.750, 127.875 | `tables/degradation_grid.csv`, worst point filter: `drop_rate=0.4`, `jitter=0.02` |
| 0.253 | `tables/degradation_grid.csv`, worst-point HOTA gap rule: `ByteTrack HOTA - Base HOTA` |
| 63.274, 79.000, 130.875 | `tables/gating_sensitivity.csv`, rows `Base` and `+gating` |
| 4.795, 4.002, 569, 372, 1.741, 0.107 | `stratified/stratified_metrics_val.csv`, bucket filters (`turning (mid)`, `low-confidence (mid)`, `occlusion (low/high)`) and differences |
| 2.375, 2.371, 2.850, 2.916, 2.906 | `tables/count_metrics_val.csv`, `row_type=mean` |
| 2.157, 3.355, 95.740, 734.211, 733.199, 69.488 | `runtime/runtime_profile.csv`, rows Base/+gating/+traj/+adaptive |

## 11) Figure/table redraw integrity

- All regenerated figures/tables in `paper/cea_draft/figs` and `paper/cea_draft/tables` are formatting-only redraws from frozen CSV sources.
- Numerical values were not edited manually; every value remains traceable to `RESULTS_ROOT` CSV files listed above.
