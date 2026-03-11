@echo off
setlocal EnableExtensions

set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%"
set "REPO_ROOT=%CD%"
set "PY=%REPO_ROOT%\.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo [ERR] venv python not found: "%PY%"
  echo       Please create venv at "%REPO_ROOT%\.venv" or update PY path.
  popd
  exit /b 1
)

if not defined SPLIT set "SPLIT=val_half"
if not defined MAX_FRAMES set "MAX_FRAMES=1000"
if not defined METHODS set "METHODS=bytetrack,ocsort"
if not defined MAX_GT_IDS set "MAX_GT_IDS=0"
if not defined SEED set "SEED=0"
if not defined DET_SOURCE set "DET_SOURCE=auto"
if not defined DROP_RATE set "DROP_RATE=0.2"
if not defined JITTER set "JITTER=0.02"
if not defined PREPARE set "PREPARE=yes"
if not defined EXISTING_MAIN_CSV set "EXISTING_MAIN_CSV=results\main_table_val_seedmean.csv"
if not defined EXISTING_SEED_ROOT set "EXISTING_SEED_ROOT=results\main_val\seed_runs"
if not defined EXISTING_SEED set "EXISTING_SEED=0"

"%PY%" scripts\run_strong_baselines.py ^
  --split %SPLIT% ^
  --max-frames %MAX_FRAMES% ^
  --methods %METHODS% ^
  --max-gt-ids %MAX_GT_IDS% ^
  --seed %SEED% ^
  --det-source %DET_SOURCE% ^
  --drop-rate %DROP_RATE% ^
  --jitter %JITTER% ^
  --prepare %PREPARE% ^
  --existing-main-csv %EXISTING_MAIN_CSV% ^
  --existing-seed-root %EXISTING_SEED_ROOT% ^
  --existing-seed %EXISTING_SEED%
if errorlevel 1 goto :error

echo Done.
echo - Combined CSV: results\main_table_val_with_baselines.csv
echo - Combined PNG: results\paper_assets_val\main_table_with_baselines.png
echo - Combined TEX: results\paper_assets_val\main_table_with_baselines.tex
echo - Strong pred root: results\strong_baselines\
popd
exit /b 0

:error
echo run_strong_baselines_val failed.
popd
exit /b 1
