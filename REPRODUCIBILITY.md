# Reproducibility

This note is intentionally narrow. It describes how to reproduce the released paper artifacts from the current repository layout without claiming that every local dataset copy or checkpoint is redistributable.

## Scope

The commands below cover the main paper-facing artifacts:

1. main table
2. degradation figure
3. runtime figure
4. gating sensitivity figure
5. optional drift operating-point figure, if the timeline CSV is included in the release

The frozen experiment spec is `results/main_val/run_config.json`. If you want paper-consistent paths and parameters, do not edit that file.

The provenance chain intentionally avoids a circular nested hash: `run_config.json` stores the resolved SHA256 of `results/main_val/release/manifest.json`, while `manifest.json` records `run_config.json` by path only rather than hashing that file again inside the manifest body. This keeps the audit trail stable and checkable.

## Tested Environment

- OS: Windows 10/11
- Python: 3.11.9
- Shell: PowerShell or `cmd.exe`
- Core dependencies: `requirements.txt`

The repository is Windows-first for the top-level `.bat` entry points. The underlying Python scripts can still be called directly on other platforms, but the exact commands below assume Windows paths.
When available, the generated `reproduce.bat` entrypoints prefer `.\.venv\Scripts\python.exe` so that the recorded Python version matches the documented environment.

## Before You Start

1. Create a virtual environment at the repository root so that the `.bat` scripts resolve `.\.venv\Scripts\python.exe` correctly.
2. Install the pinned dependencies.
3. Confirm that the required local inputs exist.

Recommended setup:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Required local assets for the core runs:

- Prepared MOT-style data root at `data/mft25_mot_full`
- Trajectory checkpoint at `runs/traj_encoder/traj_encoder.pt`

If `data/mft25_mot_full` is not already present, the main-table batch script can rebuild it only when you also have local access to the underlying source data under `data/`.

## Recommended Run Order

### 1. Main Table

Command:

```powershell
cmd /c scripts\run_main_table_val.bat
```

Primary outputs:

- `results/main_val/run_config.json`
- `results/main_val/tables/main_table_val_seedmean.csv`
- `results/main_val/tables/main_table_val_seedstd.csv`
- `results/main_val/tables/per_seq_main_val.csv`
- `results/main_val/tables/count_metrics_val.csv`
- `results/main_val/paper_assets/main_table_val_seedmean_std.png`
- `results/main_val/paper_assets/count_stability_bar.png`

Notes:

- This step also writes the official `run_config.json`.
- The script expects `runs/traj_encoder/traj_encoder.pt`. If it is missing, it attempts to train it.
- The script is the correct starting point for the rest of the release pipeline.

### 2. Degradation Figure

Default paper-compatible drop/jitter figure:

```powershell
python scripts\run_degradation_grid.py --run-config results\main_val\run_config.json
```

Primary outputs:

- `results/main_val/tables/degradation_grid.csv`
- `results/main_val/paper_assets/degradation_grid.png`
- `results/main_val/paper_assets/degradation_grid.tex`

If the release also includes the extended deployment stress figure added later, run:

```powershell
python scripts\run_degradation_extended.py --run-config results\main_val\run_config.json
```

Extended outputs:

- `results/main_val/tables/degradation_extended.csv`
- `results/main_val/paper_assets/degradation_extended.png`
- `results/main_val/paper_assets/degradation_extended_delta.tex`
- `results/main_val/paper_assets/degradation_extended_examples.png`

### 3. Runtime Figure

Command:

```powershell
python scripts\profile_runtime.py --run-config results\main_val\run_config.json
```

Primary outputs:

- `results/main_val/runtime/runtime_profile.csv`
- `results/main_val/paper_assets/runtime_profile.png`

Notes:

- `psutil` is required for this step.
- Runtime numbers are machine-dependent. Reproducing the script and the relative ordering is more realistic than expecting identical FPS on another host.

### 4. Gating Sensitivity Figure

Command:

```powershell
python scripts\run_gating_thresh_sensitivity.py --run-config results\main_val\run_config.json
```

Primary outputs:

- `results/main_val/tables/gating_sensitivity.csv`
- `results/main_val/paper_assets/gating_sensitivity.png`
- `results/main_val/paper_assets/gating_sensitivity.tex`

### 5. Optional Drift Operating-Point Figure

This figure is only reproducible if the compact timeline CSV is included in the release. It does not rerun raw drift experiments; it scans operating points from a precomputed timeline.

Command:

```powershell
python scripts\eval_drift_opscan.py --timeline-csv results\brackishmot_drift_eval_timeline.csv
```

Primary outputs:

- `results/brackishmot_drift_opscan.csv`
- `results/brackishmot_drift_selected.json`
- `paper/cea_draft/figures/fig_drift_opscan_tradeoff.pdf`
- `paper/cea_draft/tables/drift_opscan_topk.tex`

If `results/brackishmot_drift_eval_timeline.csv` is not part of a public package, this step should be documented as unavailable rather than silently skipped.

### 6. Optional Release Manifest Refresh

After the tables and figures above exist, you can refresh the paper-facing asset bundle and release metadata:

```powershell
python scripts\make_paper_assets.py --run-config results\main_val\run_config.json
```

Primary outputs:

- `results/main_val/paper_assets/paper_main_table.csv`
- `results/main_val/paper_assets/paper_main_table.tex`
- `results/main_val/paper_assets/paper_count_table.csv`
- `results/main_val/paper_assets/paper_count_table.tex`
- `results/main_val/release/manifest.json`
- `results/main_val/release/reproduce.bat`

## Minimal Release Checklist

- Keep `results/main_val/run_config.json` unchanged.
- Include a filled manifest derived from `manifest.template.json`.
- Record whether `data/mft25_mot_full` is released, withheld, or available only on request.
- Record whether `runs/traj_encoder/traj_encoder.pt` is released, withheld, or must be regenerated locally.
- Do not mark raw videos, prepared MOT exports, or model weights as public unless the release package actually contains them and their licenses permit redistribution.

## Honest Limitations

- The repository contains local data folders and checkpoints, but their presence in a private working tree does not imply public redistribution rights.
- Some figure-generation scripts can rerender outputs from released CSVs even when the raw data are unavailable; this is still useful for audit, but it is not the same as a full end-to-end rerun.
- The drift operating-point scan depends on a precomputed timeline CSV. Without that file, the scan itself is not reproducible from a code-only package.
