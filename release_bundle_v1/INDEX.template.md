# `release_bundle_v1` Index Template

Use this file as the reviewer-facing front page for the frozen bundle.

## Bundle Identity

- `release_bundle_id`: `{{ release_bundle_v1 }}`
- `generated_utc`: `{{ generated_utc }}`
- `git_commit`: `{{ git_commit }}`
- `manifest_hash`: `{{ sha256_of_results_manifest }}`

## C1: Audit-Ready Reproduction Protocol

`results/main_val/run_config.json` is the frozen experiment specification for the manuscript. Reviewers should read it together with `results/main_val/release/manifest.json` and `sha256_manifest.txt`. The intent is to make path changes, hidden reruns, and seed drift visible before they can alter the reported tables.

Recommended audit path:

1. Open `results/main_val/run_config.json`.
2. Confirm `manifest_hash`, `seed_locking`, `split`, `max_frames`, and `release_bundle_id`.
3. Verify the mirrored artifacts listed in `results/main_val/release/manifest.json`.

## C2: BrackishMOT Stress Awareness

The BrackishMOT assets are the field-proxy stress layer of the bundle.

- `results/brackishmot_groups.json`: audited clear / turbid-low / turbid-high slice selection
- `results/brackishmot_stress_metrics.csv`: natural visibility stress summary
- `results/brackishmot_drift_eval_timeline.csv`: compact drift-monitoring timeline
- `results/brackishmot_drift_opscan.csv`: threshold scan for alert operating points

Interpretation:

`clear` is the reference operating regime, while `turbid-low` and `turbid-high` expose turbidity-induced drift and sensor-jitter resilience limits before deployment.

## C3: Actionable Deployment Policy

The framework does not try to crown a universal tracker winner. It uses the measured evidence to choose the safer deployment profile.

- `paper/cea_draft/figures/fig4_deployment_decision_tree.pdf`: Base vs ByteTrack policy map
- `paper/cea_draft/tables/runtime_scopeb_brackish_true.tex`: end-to-end deployment budget evidence
- `paper/cea_draft/tables/paper_count_table.tex`: counting-stability evidence

Policy summary placeholder:

`{{ Base is default under low fragmentation + constrained compute; ByteTrack is recall reserve under persistent turbidity. }}`

## Reviewer Notes

- `{{ note_1 }}`
- `{{ note_2 }}`
- `{{ note_3 }}`
