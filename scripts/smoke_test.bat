@echo off
setlocal

set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%"

echo [1/2] Preparing train_half (1 sequence, first 200 frames)...
python scripts\prepare_mft25.py --splits train_half --seq-limit 1 --max-frames 200 --clean-split
if errorlevel 1 goto :error

echo [2/2] Running TrackEval with GT-copy predictions...
python scripts\eval_trackeval.py --split train_half
if errorlevel 1 goto :error

echo Smoke test finished successfully.
echo - GT: data\mft25_mot\train_half\^<SEQ^>\gt\gt.txt
echo - Metrics: results\metrics.csv
popd
exit /b 0

:error
echo Smoke test failed.
popd
exit /b 1
