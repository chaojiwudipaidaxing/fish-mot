# CEA MOT Paper Skeleton

Title (placeholder): **Robust Multi-Fish Tracking for Controlled Environment Aquaculture**

## 1. Introduction

- Problem context: long-term fish monitoring in controlled aquaculture tanks.
- Challenges: dense interactions, frequent occlusion, abrupt motion, and unstable detections.
- Gap: standard MOT pipelines do not directly optimize aquaculture-facing objectives (e.g., count stability).
- Contributions (placeholder):
  1. A progressive baseline stack from Base -> +gating -> +traj -> +adaptive.
  2. Aquaculture-oriented evaluation extensions (count stability + stratified difficulty analysis).
  3. Reproducible smoke-to-full pipeline with paper asset auto-generation.

## 2. Methods

### 2.1 Base (SORT-style)

- Kalman filtering for motion prediction.
- IoU + Hungarian association.
- Track lifecycle with `min_hits` and `max_age`.

### 2.2 +gating

- Mahalanobis gating before Hungarian assignment.
- Invalid high-distance matches are blocked.

### 2.3 +traj

- Add trajectory consistency cost from learned trajectory embeddings.
- Final association cost (placeholder):
  `alpha*(1-IoU) + beta*d_maha + gamma*traj_cost`.

### 2.4 +adaptive

- Adaptive trajectory weight by track state:
  - warmup stage suppresses trajectory term;
  - re-identification stage increases trajectory weight for lost/reappearing tracks.

## 3. Experiments

### 3.1 Setup

- Dataset split: `val_half` (smoke and full settings).
- Metrics: HOTA, DetA, AssA, IDF1, IDSW, plus count stability metrics.

### 3.2 Main comparison table

- Insert table:
  - `results/paper_assets_val/paper_main_table.tex`
- Figure (mean +- std):
  - `results/paper_assets_val/main_table_val_seedmean_std_paper.png`

### 3.3 Count stability

- Insert table:
  - `results/paper_assets_val/paper_count_table.tex`
- Figure:
  - `results/paper_assets_val/count_stability_bar_paper.png`

### 3.4 Stratified analysis

- Figure:
  - `results/paper_assets_val/stratified_metrics_val.png`
- Discuss bucket-specific gains/losses:
  - occlusion / density / turn / low-confidence.

### 3.5 Runtime/resource trade-offs

- Figure:
  - `results/paper_assets_val/runtime_profile.png`
- Table data:
  - `results/runtime_profile.csv`

## 4. Discussion

- Practical implications for CEA farm deployment.
- Failure cases and robustness boundaries.
- Why improvements may differ across buckets and density regimes.
- Engineering trade-off: accuracy vs speed vs memory.

## 5. Data Availability

- Data source path in this project:
  - `data/mft25_raw`
- Prepared MOT-format path:
  - `data/mft25_mot`
- Reproducibility artifacts:
  - `release/manifest.json`
  - `release/reproduce.bat`

