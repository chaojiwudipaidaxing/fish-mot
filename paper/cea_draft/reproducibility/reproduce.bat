@echo off
setlocal EnableExtensions
set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT="
for %%I in ("%SCRIPT_DIR%." "%SCRIPT_DIR%.." "%SCRIPT_DIR%..\.." "%SCRIPT_DIR%..\..\.." "%SCRIPT_DIR%..\..\..\.." "%SCRIPT_DIR%..\..\..\..\..") do (
  if not defined REPO_ROOT if exist "%%~fI\scripts\make_paper_assets.py" set "REPO_ROOT=%%~fI"
)
if not defined REPO_ROOT (
  echo [ERR] Could not locate repository root from "%SCRIPT_DIR%".
  exit /b 1
)
pushd "%REPO_ROOT%"
set "PY_EXE=python"
if exist "%REPO_ROOT%\.venv\Scripts\python.exe" set "PY_EXE=%REPO_ROOT%\.venv\Scripts\python.exe"
"%PY_EXE%" scripts\make_paper_assets.py --main-mean results\main_val\tables\main_table_val_seedmean.csv --main-std results\main_val\tables\main_table_val_seedstd.csv --count-csv results\main_val\tables\count_metrics_val.csv --strong-mean results\main_val\tables\strong_baselines_seedmean.csv --strong-std results\main_val\tables\strong_baselines_seedstd.csv --with-baselines-csv results\main_val\tables\main_table_val_with_baselines.csv --run-config results\main_val\run_config.json --out-dir results\main_val\paper_assets --release-dir results\main_val\release
if errorlevel 1 (
  echo make_paper_assets failed.
  popd
  exit /b 1
)
:: === Degradation Grid (auto) START ===
"%PY_EXE%" scripts\run_degradation_grid.py --run-config results\main_val\run_config.json
if errorlevel 1 (
  echo run_degradation_grid failed.
  popd
  exit /b 1
)
:: === Degradation Grid (auto) END ===
:: === Gating Threshold Sensitivity (auto) START ===
"%PY_EXE%" scripts\run_gating_thresh_sensitivity.py --run-config results\main_val\run_config.json
if errorlevel 1 (
  echo run_gating_thresh_sensitivity failed.
  popd
  exit /b 1
)
:: === Gating Threshold Sensitivity (auto) END ===
:: === Degradation Extended (auto) START ===
"%PY_EXE%" scripts\run_degradation_extended.py --run-config results\main_val\run_config.json
if errorlevel 1 (
  echo run_degradation_extended failed.
  popd
  exit /b 1
)
:: === Degradation Extended (auto) END ===
echo Done.
popd
exit /b 0
