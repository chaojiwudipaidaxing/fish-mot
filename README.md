# Audit-Ready Evaluation Framework for Fish MOT Reliability

This repository accompanies the revised manuscript, *A drift-aware and deployment-oriented evaluation framework for fish multi-object tracking in digital aquaculture*. It is not positioned as a tracker zoo. It is an audit-ready evaluation framework built to answer a practical agricultural question: which MOT profile can still be trusted when pond visibility shifts and counting logic starts to drift away from reality?

## C1: Frozen Experiment Specification

The repository treats `run_config.json` as the frozen experiment specification for the paper, not as a casual settings dump. The file in `results/main_val/run_config.json` records the audited split, frame budget, seeds, degradation settings, and release-bundle identity, while `manifest_hash` and `seed_locking` turn that file into a reproducibility contract. In practice, the manifest hash is the user-facing guarantee that the experiment being rerun is the same one that produced the paper tables, and the mirrored copy in `release_bundle_v1/results/main_val/run_config.json` preserves that guarantee inside the reviewer bundle.

## C2: BrackishMOT Stress Awareness

BrackishMOT is the field-proxy stress layer of the framework. Its clear, turbid-low, and turbid-high slices expose turbidity-induced drift before it silently corrupts fish counting logic, while the paired profiling and drift scripts make sensor-jitter resilience measurable rather than anecdotal. The main entry points are `scripts/select_brackish_groups.py` for audited slice selection, `scripts/eval_drift_loop.py` and `scripts/eval_drift_opscan.py` for drift monitoring, and `scripts/run_scopeb_profile_brackish_true_e2e.py` for end-to-end deployment bottleneck profiling.

## C3: Actionable Deployment Policy

The framework is designed to close the trust bridge between laboratory benchmarks and pond-side operation. The deployment decision tree in `paper/cea_draft/figures/fig4_deployment_decision_tree.pdf` translates HOTA, IDSW, runtime, memory, and drift alerts into a farm-facing policy: keep Base as the default profile when identity continuity and compute discipline matter most, and switch to ByteTrack only when repeated turbid-high evidence shows that additional recall is worth the extra fragmentation. The point is not to name a champion tracker, but to give farmers and agricultural robots a safer default under changing water conditions.

## Release Bundle Architecture

The repository keeps a working results tree and a mirrored reviewer package. `results/` stores the active experiment outputs, `paper/cea_draft/` holds manuscript source and figure/table assets, and `release_bundle_v1/` is the frozen reviewer-facing mirror. The bundle index template at `release_bundle_v1/INDEX.template.md` is intended to make that mirror legible to reviewers without forcing them to reverse-engineer the directory tree.

## Key Entry Points

- `results/main_val/run_config.json`
- `results/main_val/release/manifest.json`
- `results/brackishmot_drift_eval_timeline.csv`
- `paper/cea_draft/main.tex`
- `release_bundle_v1/INDEX.template.md`

## Route-B Positioning

This project follows a deliberate Route-B positioning strategy. In agricultural edge AI, more complex association logic is not progress by default. If a variant spends more compute, increases ID fragmentation, or weakens auditability without improving the pond-side operating point, the framework treats that as evidence of diminishing returns rather than a near-miss success. Its job is to reveal when Base is the more agriculturally robust choice, and to justify escalation only when the pond-side evidence actually repays the extra complexity.
