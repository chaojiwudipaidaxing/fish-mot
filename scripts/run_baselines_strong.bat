@echo off
setlocal

set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%"

if not defined SPLIT set "SPLIT=val_half"
if not defined MAX_FRAMES set "MAX_FRAMES=1000"
if not defined METHODS set "METHODS=bytetrack,ocsort,botsort"
if not defined MAX_GT_IDS set "MAX_GT_IDS=0"
if not defined DET_SOURCE set "DET_SOURCE=auto"
if not defined DROP_RATE set "DROP_RATE=0.0"
if not defined JITTER set "JITTER=0.0"
if not defined SEED set "SEED=0"
if not defined PREPARE set "PREPARE=yes"

python scripts\run_baselines_strong.py ^
  --split %SPLIT% ^
  --max-frames %MAX_FRAMES% ^
  --methods %METHODS% ^
  --max-gt-ids %MAX_GT_IDS% ^
  --det-source %DET_SOURCE% ^
  --drop-rate %DROP_RATE% ^
  --jitter %JITTER% ^
  --seed %SEED% ^
  --prepare %PREPARE%
if errorlevel 1 goto :error

echo Done.
echo - Baseline table: results\main_table_val_baselines.csv
echo - Baseline plot:  results\paper_assets_val\main_table_val_baselines.png
echo - Pred root:      results\baselines\
popd
exit /b 0

:error
echo run_baselines_strong failed.
popd
exit /b 1

