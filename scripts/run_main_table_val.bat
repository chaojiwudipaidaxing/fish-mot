@echo off
setlocal EnableExtensions EnableDelayedExpansion

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

if not defined MAX_FRAMES set "MAX_FRAMES=1000"
if not defined MAX_GT_IDS set "MAX_GT_IDS=50000"
if not defined DROP_RATE set "DROP_RATE=0.2"
if not defined JITTER set "JITTER=0.02"
if not defined ALPHA set "ALPHA=1.0"
if not defined BETA set "BETA=0.02"
if not defined GAMMA set "GAMMA=0.5"
if not defined ADAPTIVE_GAMMA_MIN set "ADAPTIVE_GAMMA_MIN=0.5"
if not defined ADAPTIVE_GAMMA_MAX set "ADAPTIVE_GAMMA_MAX=2.0"
if not defined ADAPTIVE_GAMMA_BOOST set "ADAPTIVE_GAMMA_BOOST=1.5"
if not defined FRAME_STATS set "FRAME_STATS=off"
if not defined SEEDS set "SEEDS=0,1,2"
if not defined CLEAN_STALE_ARTIFACTS set "CLEAN_STALE_ARTIFACTS=yes"
if not defined RESULT_NAME set "RESULT_NAME=main_val"

set "RESULT_ROOT=results\%RESULT_NAME%"
if %MAX_FRAMES% LSS 1000 set "RESULT_ROOT=results\_smoke\%RESULT_NAME%"
set "MOT_ROOT=data\mft25_mot_full"
if %MAX_FRAMES% LSS 1000 set "MOT_ROOT=data\mft25_mot_smoke"
set "OUT_ROOT=%RESULT_ROOT%"
set "SEED_ROOT=%OUT_ROOT%\seed_runs"
set "TABLE_DIR=%OUT_ROOT%\tables"
set "PAPER_DIR=%OUT_ROOT%\paper_assets"
set "RELEASE_DIR=%OUT_ROOT%\release"
set "RUN_CONFIG=%OUT_ROOT%\run_config.json"

if /I "%CLEAN_STALE_ARTIFACTS%"=="yes" (
  echo [pre] Clean stale merged/assets in %OUT_ROOT% ...
  if exist "%TABLE_DIR%\main_table_val_with_baselines.csv" del /f /q "%TABLE_DIR%\main_table_val_with_baselines.csv" >nul 2>nul
  if exist "%PAPER_DIR%" rmdir /s /q "%PAPER_DIR%"
  if exist "%RELEASE_DIR%\manifest.json" del /f /q "%RELEASE_DIR%\manifest.json" >nul 2>nul
)

echo [0/5] Write run config...
"%PY%" scripts\write_run_config.py --out %RUN_CONFIG% --exp-id main_val --result-name %RESULT_NAME% --result-root %RESULT_ROOT% --mot-root %MOT_ROOT% --split val_half --max-frames %MAX_FRAMES% --seeds %SEEDS% --drop-rate %DROP_RATE% --jitter %JITTER% --bucket-mode quantile --q1 0.33 --q2 0.66 --bucket-min-samples 200 --pred-root %SEED_ROOT%\seed_0 --methods base,gating,traj,adaptive,bytetrack,ocsort,botsort
if errorlevel 1 goto :error

echo [1/5] Prepare val_half (up to %MAX_FRAMES% frames)...
"%PY%" scripts\prepare_mft25.py --run-config %RUN_CONFIG% --out-root %MOT_ROOT% --splits val_half --max-frames %MAX_FRAMES% --clean-split
if errorlevel 1 goto :error

if not exist runs\traj_encoder\traj_encoder.pt (
  echo [2/5] traj_encoder.pt not found, train it first...
  cmd /c scripts\run_train_traj.bat
  if errorlevel 1 goto :error
) else (
  echo [2/5] Found runs\traj_encoder\traj_encoder.pt
)

echo [3/5] Run all methods for each seed: %SEEDS%
for %%S in (%SEEDS:,= %) do (
  set "SEED=%%S"
  set "OUT_DIR=!SEED_ROOT!\seed_%%S"
  echo ===== Seed %%S =====

  "%PY%" scripts\run_baseline_sort.py --run-config %RUN_CONFIG% --split val_half --mot-root %MOT_ROOT% --max-frames %MAX_FRAMES% --gating off --traj off --alpha %ALPHA% --beta 0.0 --gamma 0.0 --adaptive-gamma-min %ADAPTIVE_GAMMA_MIN% --adaptive-gamma-max %ADAPTIVE_GAMMA_MAX% --drop-rate %DROP_RATE% --jitter %JITTER% --degrade-seed %%S --frame-stats %FRAME_STATS% --clean-out --out-dir !OUT_DIR!\pred_base
  if errorlevel 1 goto :error
  "%PY%" scripts\eval_trackeval_per_seq.py --run-config %RUN_CONFIG% --split val_half --mot-root %MOT_ROOT% --pred-dir !OUT_DIR!\pred_base --tracker-name val_base_s%%S --max-frames %MAX_FRAMES% --max-gt-ids %MAX_GT_IDS% --results-per-seq !OUT_DIR!\per_seq_base.csv --results-mean !OUT_DIR!\mean_base.csv
  if errorlevel 1 goto :error

  "%PY%" scripts\run_baseline_sort.py --run-config %RUN_CONFIG% --split val_half --mot-root %MOT_ROOT% --max-frames %MAX_FRAMES% --gating on --traj off --alpha %ALPHA% --beta %BETA% --gamma 0.0 --adaptive-gamma-min %ADAPTIVE_GAMMA_MIN% --adaptive-gamma-max %ADAPTIVE_GAMMA_MAX% --drop-rate %DROP_RATE% --jitter %JITTER% --degrade-seed %%S --frame-stats %FRAME_STATS% --clean-out --out-dir !OUT_DIR!\pred_gating
  if errorlevel 1 goto :error
  "%PY%" scripts\eval_trackeval_per_seq.py --run-config %RUN_CONFIG% --split val_half --mot-root %MOT_ROOT% --pred-dir !OUT_DIR!\pred_gating --tracker-name val_gating_s%%S --max-frames %MAX_FRAMES% --max-gt-ids %MAX_GT_IDS% --results-per-seq !OUT_DIR!\per_seq_gating.csv --results-mean !OUT_DIR!\mean_gating.csv
  if errorlevel 1 goto :error

  "%PY%" scripts\run_baseline_sort.py --run-config %RUN_CONFIG% --split val_half --mot-root %MOT_ROOT% --max-frames %MAX_FRAMES% --gating on --traj on --traj-encoder runs\traj_encoder\traj_encoder.pt --alpha %ALPHA% --beta %BETA% --gamma %GAMMA% --adaptive-gamma off --adaptive-gamma-min %ADAPTIVE_GAMMA_MIN% --adaptive-gamma-max %ADAPTIVE_GAMMA_MAX% --drop-rate %DROP_RATE% --jitter %JITTER% --degrade-seed %%S --frame-stats %FRAME_STATS% --clean-out --out-dir !OUT_DIR!\pred_traj
  if errorlevel 1 goto :error
  "%PY%" scripts\eval_trackeval_per_seq.py --run-config %RUN_CONFIG% --split val_half --mot-root %MOT_ROOT% --pred-dir !OUT_DIR!\pred_traj --tracker-name val_traj_s%%S --max-frames %MAX_FRAMES% --max-gt-ids %MAX_GT_IDS% --results-per-seq !OUT_DIR!\per_seq_traj.csv --results-mean !OUT_DIR!\mean_traj.csv
  if errorlevel 1 goto :error

  "%PY%" scripts\run_baseline_sort.py --run-config %RUN_CONFIG% --split val_half --mot-root %MOT_ROOT% --max-frames %MAX_FRAMES% --gating on --traj on --traj-encoder runs\traj_encoder\traj_encoder.pt --alpha %ALPHA% --beta %BETA% --gamma %GAMMA% --adaptive-gamma on --adaptive-gamma-boost %ADAPTIVE_GAMMA_BOOST% --adaptive-gamma-min %ADAPTIVE_GAMMA_MIN% --adaptive-gamma-max %ADAPTIVE_GAMMA_MAX% --drop-rate %DROP_RATE% --jitter %JITTER% --degrade-seed %%S --frame-stats %FRAME_STATS% --clean-out --out-dir !OUT_DIR!\pred_adaptive
  if errorlevel 1 goto :error
  "%PY%" scripts\eval_trackeval_per_seq.py --run-config %RUN_CONFIG% --split val_half --mot-root %MOT_ROOT% --pred-dir !OUT_DIR!\pred_adaptive --tracker-name val_adaptive_s%%S --max-frames %MAX_FRAMES% --max-gt-ids %MAX_GT_IDS% --results-per-seq !OUT_DIR!\per_seq_adaptive.csv --results-mean !OUT_DIR!\mean_adaptive.csv
  if errorlevel 1 goto :error

  "%PY%" scripts\build_main_table_val.py --base-mean !OUT_DIR!\mean_base.csv --gating-mean !OUT_DIR!\mean_gating.csv --traj-mean !OUT_DIR!\mean_traj.csv --adaptive-mean !OUT_DIR!\mean_adaptive.csv --base-per-seq !OUT_DIR!\per_seq_base.csv --gating-per-seq !OUT_DIR!\per_seq_gating.csv --traj-per-seq !OUT_DIR!\per_seq_traj.csv --adaptive-per-seq !OUT_DIR!\per_seq_adaptive.csv --out-main !OUT_DIR!\main_table_val.csv --out-per-seq !OUT_DIR!\per_seq_main_val.csv
  if errorlevel 1 goto :error
)

echo [4/5] Aggregate seed mean/std + plot...
"%PY%" scripts\aggregate_main_table_seeds.py --run-config %RUN_CONFIG% --seed-root %SEED_ROOT% --seeds %SEEDS% --out-mean %TABLE_DIR%\main_table_val_seedmean.csv --out-std %TABLE_DIR%\main_table_val_seedstd.csv --out-per-seq %TABLE_DIR%\per_seq_main_val.csv --plot-path %PAPER_DIR%\main_table_val_seedmean_std.png
if errorlevel 1 goto :error

if not defined COUNT_SEED (
  for /f "tokens=1 delims=," %%A in ("%SEEDS%") do set "COUNT_SEED=%%A"
)

echo [5/5] Evaluate count stability on seed !COUNT_SEED!...
"%PY%" scripts\eval_count_stability.py --run-config %RUN_CONFIG% --split val_half --mot-root %MOT_ROOT% --max-frames %MAX_FRAMES% --pred-root %SEED_ROOT%\seed_!COUNT_SEED! --output-csv %TABLE_DIR%\count_metrics_val.csv --paper-assets-dir %PAPER_DIR%
if errorlevel 1 goto :error

echo Done.
echo - Run config: %RUN_CONFIG%
echo - Seed mean: %TABLE_DIR%\main_table_val_seedmean.csv
echo - Seed std:  %TABLE_DIR%\main_table_val_seedstd.csv
echo - Per-seq all seeds: %TABLE_DIR%\per_seq_main_val.csv
echo - Plot: %PAPER_DIR%\main_table_val_seedmean_std.png
echo - Count metrics: %TABLE_DIR%\count_metrics_val.csv
echo - Count plot: %PAPER_DIR%\count_stability_bar.png
echo - Config: RESULT_ROOT=%RESULT_ROOT% MOT_ROOT=%MOT_ROOT% SEEDS=%SEEDS% MAX_FRAMES=%MAX_FRAMES% MAX_GT_IDS=%MAX_GT_IDS% DROP_RATE=%DROP_RATE% JITTER=%JITTER%
popd
exit /b 0

:error
echo run_main_table_val failed.
popd
exit /b 1
