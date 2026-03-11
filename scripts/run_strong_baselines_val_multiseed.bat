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
if not defined METHODS set "METHODS=bytetrack,ocsort,botsort"
if not defined MAX_GT_IDS set "MAX_GT_IDS=50000"
if not defined SEEDS set "SEEDS=0,1,2"
if not defined DET_SOURCE set "DET_SOURCE=auto"
if not defined DROP_RATE set "DROP_RATE=0.2"
if not defined JITTER set "JITTER=0.02"
if not defined ADAPTIVE_GAMMA_MIN set "ADAPTIVE_GAMMA_MIN=0.5"
if not defined ADAPTIVE_GAMMA_MAX set "ADAPTIVE_GAMMA_MAX=2.0"
if not defined FRAME_STATS set "FRAME_STATS=off"
if not defined PREPARE set "PREPARE=yes"
if not defined TRAJ_ENCODER set "TRAJ_ENCODER=runs\traj_encoder\traj_encoder.pt"
if not defined CLEAN_STALE_ARTIFACTS set "CLEAN_STALE_ARTIFACTS=yes"
if not defined RESULT_NAME set "RESULT_NAME=main_val"

set "RESULT_ROOT=results\%RESULT_NAME%"
if %MAX_FRAMES% LSS 1000 set "RESULT_ROOT=results\_smoke\%RESULT_NAME%"
set "MOT_ROOT=data\mft25_mot_full"
if %MAX_FRAMES% LSS 1000 set "MOT_ROOT=data\mft25_mot_smoke"
set "TABLE_DIR=%RESULT_ROOT%\tables"
set "PAPER_DIR=%RESULT_ROOT%\paper_assets"
set "RELEASE_DIR=%RESULT_ROOT%\release"
set "STRONG_ROOT=%RESULT_ROOT%\strong_baselines"
set "RUN_CONFIG=%RESULT_ROOT%\run_config.json"
if not defined EXISTING_MAIN_MEAN_CSV set "EXISTING_MAIN_MEAN_CSV=%TABLE_DIR%\main_table_val_seedmean.csv"
if not defined EXISTING_MAIN_STD_CSV set "EXISTING_MAIN_STD_CSV=%TABLE_DIR%\main_table_val_seedstd.csv"
if not defined EXISTING_SEED_ROOT set "EXISTING_SEED_ROOT=%RESULT_ROOT%\seed_runs"

if /I "%CLEAN_STALE_ARTIFACTS%"=="yes" (
  echo [pre] Clean stale strong merged/assets in %RESULT_ROOT% ...
  if exist "%TABLE_DIR%\main_table_val_with_baselines.csv" del /f /q "%TABLE_DIR%\main_table_val_with_baselines.csv" >nul 2>nul
  if exist "%PAPER_DIR%\main_table_with_baselines.png" del /f /q "%PAPER_DIR%\main_table_with_baselines.png" >nul 2>nul
  if exist "%PAPER_DIR%\main_table_with_baselines.tex" del /f /q "%PAPER_DIR%\main_table_with_baselines.tex" >nul 2>nul
  if exist "%PAPER_DIR%\strong_baselines_seedmean_std.png" del /f /q "%PAPER_DIR%\strong_baselines_seedmean_std.png" >nul 2>nul
  if exist "%PAPER_DIR%\strong_baselines_seedmean_std.tex" del /f /q "%PAPER_DIR%\strong_baselines_seedmean_std.tex" >nul 2>nul
  if exist "%RELEASE_DIR%\manifest.json" del /f /q "%RELEASE_DIR%\manifest.json" >nul 2>nul
)

"%PY%" scripts\write_run_config.py --out %RUN_CONFIG% --exp-id strong_baselines --result-name %RESULT_NAME% --result-root %RESULT_ROOT% --mot-root %MOT_ROOT% --split %SPLIT% --max-frames %MAX_FRAMES% --seeds %SEEDS% --drop-rate %DROP_RATE% --jitter %JITTER% --bucket-mode quantile --q1 0.33 --q2 0.66 --bucket-min-samples 200 --pred-root %RESULT_ROOT%\seed_runs\seed_0 --methods %METHODS%
if errorlevel 1 goto :error

"%PY%" scripts\run_strong_baselines.py ^
  --run-config %RUN_CONFIG% ^
  --split %SPLIT% ^
  --mot-root %MOT_ROOT% ^
  --max-frames %MAX_FRAMES% ^
  --methods %METHODS% ^
  --max-gt-ids %MAX_GT_IDS% ^
  --seeds %SEEDS% ^
  --det-source %DET_SOURCE% ^
  --drop-rate %DROP_RATE% ^
  --jitter %JITTER% ^
  --adaptive-gamma-min %ADAPTIVE_GAMMA_MIN% ^
  --adaptive-gamma-max %ADAPTIVE_GAMMA_MAX% ^
  --frame-stats %FRAME_STATS% ^
  --prepare %PREPARE% ^
  --output-root %STRONG_ROOT% ^
  --traj-encoder %TRAJ_ENCODER% ^
  --existing-main-mean-csv %EXISTING_MAIN_MEAN_CSV% ^
  --existing-main-std-csv %EXISTING_MAIN_STD_CSV% ^
  --existing-seed-root %EXISTING_SEED_ROOT% ^
  --strong-mean-out %TABLE_DIR%\strong_baselines_seedmean.csv ^
  --strong-std-out %TABLE_DIR%\strong_baselines_seedstd.csv ^
  --strong-plot-out %PAPER_DIR%\strong_baselines_seedmean_std.png ^
  --strong-tex-out %PAPER_DIR%\strong_baselines_seedmean_std.tex ^
  --table-out %TABLE_DIR%\main_table_val_with_baselines.csv ^
  --plot-out %PAPER_DIR%\main_table_with_baselines.png ^
  --tex-out %PAPER_DIR%\main_table_with_baselines.tex
if errorlevel 1 goto :error

echo Done.
echo - Run config:       %RUN_CONFIG%
echo - Strong seed mean: %TABLE_DIR%\strong_baselines_seedmean.csv
echo - Strong seed std:  %TABLE_DIR%\strong_baselines_seedstd.csv
echo - Strong plot:      %PAPER_DIR%\strong_baselines_seedmean_std.png
echo - Strong tex:       %PAPER_DIR%\strong_baselines_seedmean_std.tex
echo - Main table:       %TABLE_DIR%\main_table_val_with_baselines.csv
echo - Main plot:        %PAPER_DIR%\main_table_with_baselines.png
echo - Main tex:         %PAPER_DIR%\main_table_with_baselines.tex
echo - Config: RESULT_ROOT=%RESULT_ROOT% MOT_ROOT=%MOT_ROOT% METHODS=%METHODS% SEEDS=%SEEDS% MAX_FRAMES=%MAX_FRAMES% PREPARE=%PREPARE%
popd
exit /b 0

:error
echo run_strong_baselines_val_multiseed failed.
popd
exit /b 1
