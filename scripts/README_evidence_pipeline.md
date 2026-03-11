# Evidence Pipeline Entry Scan and Next Scripts

## 1) Existing evaluation/export script entry points (keyword scan)

- `scripts/profile_runtime.py`
  - Runtime/resource profiling for `Base/+gating/+traj/+adaptive`.
- `scripts/eval_trackeval.py`
  - TrackEval wrapper (global split-level MOT evaluation).
- `scripts/eval_trackeval_per_seq.py`
  - TrackEval wrapper (per-sequence evaluation + aggregate outputs).
- `scripts/run_degradation_grid.py`
  - Controlled degradation grid runner (`drop/jitter`) for robustness plots/tables.
- `scripts/run_gating_thresh_sensitivity.py`
  - Gating-threshold sensitivity sweep and plotting.
- `scripts/build_ablation_gating.py`
  - Off/on gating ablation CSV builder.
- `scripts/format_elsevier_word.py`
  - Word post-processing (styles, table cleanup, reference cleanup, manuscript packaging).
- `scripts/build_cea_word_release.sh`
  - End-to-end LaTeX→Word release build pipeline.
- `scripts/export_pdf_with_chrome.ps1`
  - HTML→PDF fallback export via Chrome headless.
- `scripts/validate_cea_outputs.py`
  - CEA Word-package validation gates.
- `scripts/validate_compag_strict.py`
  - Strict COMPAG/CEA validation gates for final deliverables.

## 2) Manuscript path and figure/table output directories

- Main manuscript source:
  - `paper/cea_draft/main.tex`
- Active figure directory (exists):
  - `paper/cea_draft/figs`
- Active table directory (exists):
  - `paper/cea_draft/tables`
- `main.tex` currently references figure/table assets via:
  - `\includegraphics{figs/...}` and `\input{tables/...}`.

## 3) Current unresolved placeholder references in `paper/cea_draft/main.tex`

- Gating activation planned placeholders:
  - `Fig~X` at `paper/cea_draft/main.tex:330`
  - `Table~Y` at `paper/cea_draft/main.tex:331`
  - `Fig~Z` at `paper/cea_draft/main.tex:332`
- Stress-test table location:
  - Stress section heading at `paper/cea_draft/main.tex:138`
  - Stress table caption at `paper/cea_draft/main.tex:149`
  - Stress table label `tab:stress_param_template` at `paper/cea_draft/main.tex:150`
- Scope B runtime schema location:
  - Scope A/B paragraph at `paper/cea_draft/main.tex:179`
  - Scope B schema caption at `paper/cea_draft/main.tex:378`
  - Scope B schema label `tab:runtime_e2e_fields` at `paper/cea_draft/main.tex:379`
  - Scope B TODO execution note at `paper/cea_draft/main.tex:397`

## 4) New scripts to add next (name + responsibility)

1. `scripts/run_scopeb_profile.py`
   - Execute Scope B (end-to-end) profiling under fixed protocol.
   - Output CSV fields (exact names): `fps_e2e`, `mem_peak_mb_e2e`, `cpu_norm_e2e`, `decode_time`, `detector_time`, `tracking_time`, `write_time`, `detector_name`, `input_resolution`.

2. `scripts/run_env_stress_tests.py`
   - Run environment-aware stress matrix (e.g., motion blur / low-light / turbidity) with 3-level strengths.
   - Emit reproducible parameter log and metric-by-strength CSV for paper figures/tables.

3. `scripts/eval_gating_activation.py`
   - Compute gating activation diagnostics from run logs:
     - trigger rate (frame/sequence),
     - trigger event-length distribution,
     - gating score CDF with threshold markers.
   - Export `Fig X`, `Table Y`, `Fig Z` assets for manuscript insertion.

4. `scripts/eval_drift_loop.py`
   - Evaluate drift monitoring loop (`D_in(t)`, `D_out(t)`, `tau_in`, `tau_out`, `K`) on frozen logs.
   - Export alert/fallback timeline summaries and a reproducible drift-loop audit CSV.

## 5) Initial implementation status (scaffold created)

The following script files are now created as runnable scaffolds (template outputs only, no fabricated metrics):

- `scripts/run_scopeb_profile.py`
- `scripts/run_env_stress_matrix.py`
- `scripts/eval_gating_activation.py`
- `scripts/eval_drift_loop.py`

The full stress-test runner used for execution is:

- `scripts/run_env_stress_tests.py`
