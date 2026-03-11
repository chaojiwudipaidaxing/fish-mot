@echo off
setlocal

set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%"

echo [1/6] Prepare MOT data (train_half, 1 sequence, 200 frames)...
python scripts\prepare_mft25.py --splits train_half --seq-limit 1 --max-frames 200 --clean-split
if errorlevel 1 goto :error

echo [2/6] Run baseline SORT (gating off)...
python scripts\run_baseline_sort.py --split train_half --max-frames 200 --gating off --clean-out --out-dir results\baseline\pred_off
if errorlevel 1 goto :error

echo [3/6] Evaluate gating-off predictions...
python scripts\eval_trackeval.py --split train_half --tracker-name gating_off --pred-dir results\baseline\pred_off --results-csv results\baseline\metrics_gating_off.csv --trackers-root results\baseline\trackeval_off
if errorlevel 1 goto :error

echo [4/6] Run baseline SORT (gating on)...
python scripts\run_baseline_sort.py --split train_half --max-frames 200 --gating on --gating-thresh 1.0 --clean-out --out-dir results\baseline\pred_on
if errorlevel 1 goto :error

echo [5/6] Evaluate gating-on predictions...
python scripts\eval_trackeval.py --split train_half --tracker-name gating_on --pred-dir results\baseline\pred_on --results-csv results\baseline\metrics_gating_on.csv --trackers-root results\baseline\trackeval_on
if errorlevel 1 goto :error

echo [6/6] Build gating ablation table...
python scripts\build_ablation_gating.py --off-csv results\baseline\metrics_gating_off.csv --on-csv results\baseline\metrics_gating_on.csv --out-csv results\ablation_gating.csv
if errorlevel 1 goto :error

echo P2 pipeline finished successfully.
echo - Ablation: results\ablation_gating.csv
popd
exit /b 0

:error
echo P2 pipeline failed.
popd
exit /b 1
