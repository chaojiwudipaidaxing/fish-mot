# Graphical Abstract Plan (CEA)

## Target specification
- One-panel graphic, Elsevier-friendly.
- Suggested size: 1328 x 531 px (width x height), PNG or TIFF.
- Keep labels readable after reduction (>=8 pt equivalent).

## Storyline (left -> right)
1. Reproducibility lock:
   - run_config.json, mot_root isolation, manifest hashes.
2. Evaluation stack:
   - main table + strong baselines, count stability, stratified buckets.
3. Runtime and deployment:
   - FPS/memory/CPU chart and rule-based method selection.
4. Controlled degradation block:
   - protocol card (drop/jitter grid); numeric panel enabled when frozen
     degradation_grid asset is present.

## Visual modules
- Module A: "Drift-proof reproducibility" badge with git_commit.
- Module B: metric bars (HOTA/IDF1/IDSW) for Base and baselines.
- Module C: stratified heat/risk panel (turning and low-confidence warnings).
- Module D: deployment decision tree by FPS and memory budget.

## Draft caption
"From locked configuration to deployment decision: a reproducible, stratified,
robustness-cost evaluation workflow for fish MOT in digital aquaculture."

## Asset sources
- figs/main_table_with_baselines.png
- figs/stratified_metrics_val.png
- figs/runtime_profile.png
- figs/count_stability_bar_paper.png
