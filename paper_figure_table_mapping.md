# Figure/Table Mapping (CEA Route B Draft)

| Name in Paper | Type | Path | Usage |
|---|---|---|---|
| Table 1. Main val_half comparison (mean+/-std) | Table | `results/main_val/tables/main_table_val_with_baselines.csv` | Main comparison without SOTA claim |
| Table 2. In-house methods seed mean | Table | `results/main_val/tables/main_table_val_seedmean.csv` | Base/+gating/+traj/+adaptive reference |
| Table 3. In-house methods seed std | Table | `results/main_val/tables/main_table_val_seedstd.csv` | Variance reporting |
| Table 4. Strong baselines seed mean | Table | `results/main_val/tables/strong_baselines_seedmean.csv` | Fair baseline evidence |
| Table 5. Strong baselines seed std | Table | `results/main_val/tables/strong_baselines_seedstd.csv` | Baseline variance reporting |
| Figure 1. Main table overview | Figure | `results/main_val/paper_assets/main_table_with_baselines.png` | Visual summary of main comparison |
| Table 6. Controlled degradation grid | Table | `results/main_val/tables/degradation_grid.csv` | Robustness under drop/jitter |
| Figure 2. Degradation robustness curves | Figure | `results/main_val/paper_assets/degradation_grid.png` | HOTA vs drop_rate by jitter and method |
| Table 7. Stratified metrics | Table | `results/main_val/stratified/stratified_metrics_val.csv` | Failure mode analysis |
| Figure 3. Stratified metrics plot | Figure | `results/main_val/paper_assets/stratified_metrics_val.png` | Bucket-level behavior |
| Table 8. Runtime profile | Table | `results/main_val/runtime/runtime_profile.csv` | Deployment compute trade-off |
| Figure 4. Runtime profile plot | Figure | `results/main_val/paper_assets/runtime_profile.png` | FPS/memory/CPU visualization |
| Table 9. Count stability metrics | Table | `results/main_val/tables/count_metrics_val.csv` | Husbandry-oriented count quality |
| Figure 5. Count stability bar | Figure | `results/main_val/paper_assets/count_stability_bar_paper.png` | Count metric comparison |
| Reproducibility manifest | Artifact | `results/main_val/release/manifest.json` | Commit + hashes + config traceability |
| Reproduce script | Artifact | `results/main_val/release/reproduce.bat` | One-click rerun (includes degradation section) |
| Paper main table (LaTeX) | TeX | `results/main_val/paper_assets/paper_main_table.tex` | Direct manuscript insertion |
| Paper count table (LaTeX) | TeX | `results/main_val/paper_assets/paper_count_table.tex` | Direct manuscript insertion |
| Degradation table (LaTeX) | TeX | `results/main_val/paper_assets/degradation_grid.tex` | Robustness subsection insertion |
