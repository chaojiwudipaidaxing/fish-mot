# Fish-MOT (Route B): Robustness and Deployment Trade-off Draft for CEA

## Positioning Statement (Route B, strictly non-SOTA)

This draft follows a robustness/trade-off narrative for deployment in aquaculture monitoring, not an overall-SOTA claim.  
All numbers below are taken directly from experiment artifacts under `results/main_val/...` (no synthetic values).

---

## 1. Introduction

Aquaculture monitoring requires trackers that are not only accurate under clean conditions, but also stable when detection quality degrades and when compute budgets are constrained. In this work, we present an engineering-oriented pipeline for fish MOT with:

1. drift-proof reproducibility (run config + manifest + gated checks),
2. controlled degradation protocol (`drop_rate`, `jitter`) for robustness analysis,
3. quantile-based stratified evaluation for failure mode diagnosis,
4. counting stability metrics for husbandry-relevant outcomes,
5. runtime/resource profiling for deployment decisions,
6. fair comparison against strong baselines under the same detection source.

---

## 2. Experimental Protocol

- Split: `val_half`, frame cap: `1000`, seeds: `0,1,2` for main/strong tables.
- Detection source for fair comparison: shared MOT-format input pipeline.
- Main methods: `Base`, `+gating`, `+traj`, `+adaptive`.
- Strong baselines: `ByteTrack`, `OC-SORT`, `BoT-SORT`.
- Reproducibility anchors:
  - `results/main_val/run_config.json`
  - `results/main_val/release/manifest.json`
  - `results/main_val/release/reproduce.bat`

---

## 3. Main Results (val_half, mean+/-std)

Source: `results/main_val/tables/main_table_val_with_baselines.csv`

| Method | HOTA | DetA | AssA | IDF1 | IDSW |
|---|---:|---:|---:|---:|---:|
| Base | 58.232 +/- 0.304 | 73.150 +/- 0.037 | 48.121 +/- 0.577 | 64.042 +/- 0.550 | 79.375 +/- 1.203 |
| +gating | 50.345 +/- 2.317 | 72.504 +/- 0.033 | 37.281 +/- 3.652 | 53.967 +/- 3.051 | 130.458 +/- 0.425 |
| +traj | 50.695 +/- 1.557 | 72.448 +/- 0.135 | 37.332 +/- 2.443 | 54.277 +/- 1.996 | 117.792 +/- 0.680 |
| +adaptive | 51.071 +/- 1.435 | 72.538 +/- 0.226 | 37.771 +/- 2.167 | 54.667 +/- 1.779 | 113.792 +/- 0.257 |
| ByteTrack | 58.431 +/- 0.291 | 73.858 +/- 0.015 | 48.049 +/- 0.569 | 64.011 +/- 0.545 | 88.375 +/- 0.872 |
| OC-SORT | 49.732 +/- 1.619 | 73.985 +/- 0.135 | 35.605 +/- 2.352 | 52.631 +/- 2.057 | 145.875 +/- 1.643 |
| BoT-SORT | 50.414 +/- 0.981 | 73.426 +/- 0.087 | 36.214 +/- 1.425 | 53.412 +/- 1.047 | 121.000 +/- 4.056 |

Interpretation (Route B):
- We do not claim overall best.
- Under this setup, `Base` and `ByteTrack` are the strongest references on aggregate HOTA/IDF1.
- Our contribution is the controlled robustness and deployment analysis pipeline, not a universal leaderboard win.

---

## 4. Robustness Under Controlled Degradation

Source: `results/main_val/tables/degradation_grid.csv`  
Figure: `results/main_val/paper_assets/degradation_grid.png`

Protocol fixed in script:
- split=`val_half`, max_frames=`1000`, seed=`0`, det_source=`auto`, gating_thresh=`2000`
- grid: `drop_rate in {0.0, 0.2, 0.4}`, `jitter in {0.0, 0.02}`
- methods: `Base`, `+gating`, `ByteTrack`

### 4.1 Degradation slope (drop 0.0 -> 0.4)

#### jitter = 0.00
- Base: DeltaHOTA = **35.707**, DeltaIDF1 = **27.959**
- +gating: DeltaHOTA = **28.299**, DeltaIDF1 = **19.257**
- ByteTrack: DeltaHOTA = **35.557**, DeltaIDF1 = **27.857**

#### jitter = 0.02
- Base: DeltaHOTA = **33.474**, DeltaIDF1 = **25.141**
- +gating: DeltaHOTA = **24.139**, DeltaIDF1 = **14.888**
- ByteTrack: DeltaHOTA = **33.382**, DeltaIDF1 = **25.082**

Observation:
- `+gating` shows a smaller drop magnitude as drop_rate increases, but from a lower absolute baseline.
- This supports a trade-off perspective: slope robustness and absolute quality should be reported together.

### 4.2 Worst-point ranking (drop=0.4)

At `jitter=0.00`:
- HOTA rank: ByteTrack (39.905) > Base (39.584) > +gating (36.303)
- IDF1 rank: ByteTrack (46.764) > Base (46.730) > +gating (42.452)
- IDSW (lower better): Base (89.750) < ByteTrack (104.125) < +gating (112.000)

At `jitter=0.02`:
- HOTA rank: ByteTrack (41.435) > Base (41.182) > +gating (38.727)
- IDF1 rank: Base (49.770) = ByteTrack (49.770) > +gating (45.989)
- IDSW (lower better): Base (99.625) < ByteTrack (112.750) < +gating (127.875)

---

## 5. Stratified Failure Analysis

Source: `results/main_val/stratified/stratified_metrics_val.csv`  
Figure: `results/main_val/paper_assets/stratified_metrics_val.png`

Bucket mode:
- quantile 33%/66% with non-empty bucket guarantees (`bucket_mode=quantile`)
- no zero-frame bucket rows in final CSV

Most challenging region (for all four in-house variants):
- **low_conf / low** bucket (`bucket_share ~ 0.330`)
  - Base: F1 80.021, IDSW 555
  - +gating: F1 79.576, IDSW 873
  - +traj: F1 79.629, IDSW 796
  - +adaptive: F1 79.678, IDSW 784

Interpretation:
- Low-confidence observations are the dominant failure mode for identity continuity.
- This directly informs deployment: camera quality and detector confidence calibration are high-priority knobs.

---

## 6. Counting Stability (Husbandry-Relevant)

Source: `results/main_val/tables/count_metrics_val.csv`  
Figure: `results/main_val/paper_assets/count_stability_bar_paper.png`

Mean rows:
- Base: CountMAE 2.321, CountRMSE 2.850, CountVar 3.746, CountDrift 2.075
- +gating: CountMAE 2.379, CountRMSE 2.916, CountVar 3.825, CountDrift 2.075
- +traj: CountMAE 2.375, CountRMSE 2.911, CountVar 3.817, CountDrift 2.083
- +adaptive: CountMAE 2.371, CountRMSE 2.906, CountVar 3.808, CountDrift 2.100

Interpretation:
- Count stability remains comparable across variants; Base is slightly better on MAE/RMSE in this setup.
- These indicators are useful for production-facing stocking/flow monitoring where absolute counts matter.

---

## 7. Discussion: Deployment Guideline (CPU / Edge Cases)

Source: `results/main_val/runtime/runtime_profile.csv`  
Figure: `results/main_val/paper_assets/runtime_profile.png`

Measured runtime/resource:
- Base: FPS 822.812, mem 548.992 MB, cpu_norm 2.157%
- +gating: FPS 1182.212, mem 548.395 MB, cpu_norm 3.355%
- +traj: FPS 95.740, mem 734.211 MB, cpu_norm 69.259%
- +adaptive: FPS 89.024, mem 733.199 MB, cpu_norm 69.488%

Threshold-style recommendation:
1. **Edge/CPU-constrained target** (>=30 FPS, <=600 MB, low CPU budget): prefer **Base** or **+gating**.
2. **Identity-sensitive but still online**: start with Base and monitor low_conf-low bucket; add gating only after threshold re-tuning validation.
3. **Server-side/offline analytics** (higher RAM/CPU available): `+traj`/`+adaptive` are feasible, but evaluate benefit vs heavy CPU/RAM cost per site.

Connection to stratified failures:
- Because low_conf-low bucket is the hardest region, deployment priority should be detector confidence quality and camera-domain calibration before adding heavy temporal modules.

---

## 8. Reproducibility & Data Availability

Sources:
- `results/main_val/release/manifest.json`
- `results/main_val/release/reproduce.bat`

Current reproducibility metadata:
- `git_commit = TESTCOMMIT1234567`
- input hashes recorded: **10**
- output hashes recorded: **9**
- `reproduce.bat` includes Degradation Grid section for end-to-end rerun.

Data and scripts are organized with:
- smoke/full isolation in MOT roots,
- run-config-driven path control,
- artifact gate checks to prevent data drift and stale table reuse.

---

## 9. CEA-Oriented Engineering Contributions (What This Paper Claims)

1. **Drift-proof reproducibility**: run_config + manifest + artifact gate for auditable experiments.
2. **Controlled degradation protocol**: explicit drop/jitter grid for robustness reporting.
3. **Stratified evaluation**: quantile buckets for occlusion/density/turn/low_conf with non-empty coverage.
4. **Counting stability metrics**: livestock-relevant CountMAE/RMSE/Drift reporting.
5. **Runtime/resource profiling**: FPS + memory + normalized CPU for deployment planning.
6. **Fair strong-baseline comparison**: same detection input and unified evaluation interface.

---

## 10. Limitations and Next Steps (Non-SOTA Direction)

- This work does not claim universal leaderboard dominance.
- Current performance emphasizes engineering robustness analysis and deployment transparency.
- Future low-cost extension: gating-threshold sensitivity mini-grid (e.g., 1000/2000/4000, seed=0) to further justify threshold choice for reviewers.
