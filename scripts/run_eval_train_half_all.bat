@echo off
setlocal

set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%"

echo [1/2] Prepare train_half (all 8 seq, up to first 1000 frames)...
python scripts\prepare_mft25.py --splits train_half --max-frames 1000 --clean-split
if errorlevel 1 goto :error

echo [2/2] Evaluate train_half per-sequence with TrackEval...
python scripts\eval_trackeval_per_seq.py --split train_half --max-frames 1000 --max-gt-ids 2000 --tracker-name train_half_per_seq --results-per-seq results\per_seq_metrics.csv --results-mean results\metrics_mean.csv
if errorlevel 1 goto :error

echo Done.
echo - Per-seq: results\per_seq_metrics.csv
echo - Mean:    results\metrics_mean.csv
popd
exit /b 0

:error
echo Per-sequence evaluation failed.
popd
exit /b 1
