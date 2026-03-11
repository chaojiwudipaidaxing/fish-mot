@echo off
setlocal

set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%"

echo [1/3] Prepare MOT data (train_half, 1 sequence, 200 frames)...
python scripts\prepare_mft25.py --splits train_half --seq-limit 1 --max-frames 200 --clean-split
if errorlevel 1 goto :error

echo [2/3] Run baseline SORT tracker...
python scripts\run_baseline_sort.py --split train_half --max-frames 200 --clean-out --out-dir results\baseline\pred
if errorlevel 1 goto :error

echo [3/3] Evaluate baseline predictions with TrackEval...
python scripts\eval_trackeval.py --split train_half --tracker-name baseline_sort --pred-dir results\baseline\pred --results-csv results\baseline\metrics.csv --trackers-root results\baseline\trackeval
if errorlevel 1 goto :error

echo P1 pipeline finished successfully.
echo - Pred: results\baseline\pred\^<SEQ^>.txt
echo - Metrics: results\baseline\metrics.csv
popd
exit /b 0

:error
echo P1 pipeline failed.
popd
exit /b 1
