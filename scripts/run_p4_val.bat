@echo off
setlocal

set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%"

set "MAX_FRAMES=1000"
set "DROP_RATE=0.2"
set "JITTER=0.02"

echo [1/7] Prepare train_half (8 sequences, first %MAX_FRAMES% frames)...
python scripts\prepare_mft25.py --splits train_half --max-frames %MAX_FRAMES% --clean-split
if errorlevel 1 goto :error

if not exist runs\traj_encoder\traj_encoder.pt (
  echo [2/7] traj_encoder.pt not found, train it first...
  cmd /c scripts\run_train_traj.bat
  if errorlevel 1 goto :error
) else (
  echo [2/7] Found runs\traj_encoder\traj_encoder.pt
)

echo [3/7] Generate Base hard predictions...
python scripts\run_baseline_sort.py --split train_half --max-frames %MAX_FRAMES% --gating off --traj off --alpha 1.0 --beta 0.0 --gamma 0.0 --drop-rate %DROP_RATE% --jitter %JITTER% --clean-out --out-dir results\p4\pred_base_hard
if errorlevel 1 goto :error

echo [4/7] Generate +gating hard predictions...
python scripts\run_baseline_sort.py --split train_half --max-frames %MAX_FRAMES% --gating on --traj off --alpha 1.0 --beta 0.02 --gamma 0.0 --drop-rate %DROP_RATE% --jitter %JITTER% --clean-out --out-dir results\p4\pred_gating_hard
if errorlevel 1 goto :error

echo [5/7] Generate +traj hard predictions...
python scripts\run_baseline_sort.py --split train_half --max-frames %MAX_FRAMES% --gating on --traj on --traj-encoder runs\traj_encoder\traj_encoder.pt --alpha 1.0 --beta 0.02 --gamma 0.5 --adaptive-gamma off --drop-rate %DROP_RATE% --jitter %JITTER% --clean-out --out-dir results\p4\pred_traj_hard
if errorlevel 1 goto :error

echo [6/7] Generate +adaptive hard predictions...
python scripts\run_baseline_sort.py --split train_half --max-frames %MAX_FRAMES% --gating on --traj on --traj-encoder runs\traj_encoder\traj_encoder.pt --alpha 1.0 --beta 0.02 --gamma 0.5 --adaptive-gamma on --adaptive-gamma-boost 1.5 --drop-rate %DROP_RATE% --jitter %JITTER% --clean-out --out-dir results\p4\pred_adaptive_hard
if errorlevel 1 goto :error

echo [7/7] Stratified eval + paper assets...
python scripts\eval_stratified_metrics.py --split train_half --max-frames %MAX_FRAMES% --max-gt-ids 2000 --pred-base results\p4\pred_base_hard --pred-gating results\p4\pred_gating_hard --pred-traj results\p4\pred_traj_hard --pred-adaptive results\p4\pred_adaptive_hard --output-csv results\stratified_metrics.csv --paper-assets-dir results\paper_assets
if errorlevel 1 goto :error

echo Done.
echo - Stratified metrics: results\stratified_metrics.csv
echo - Paper assets dir:   results\paper_assets
popd
exit /b 0

:error
echo P4 pipeline failed.
popd
exit /b 1
