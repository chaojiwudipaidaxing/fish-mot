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

echo [1/2] Prepare MOT data (train_half, BT-001, first 2000 frames)...
"%PY%" scripts\prepare_mft25.py --splits train_half --seq-limit 1 --max-frames 2000 --clean-split
if errorlevel 1 goto :error

echo [2/2] Train trajectory encoder (1D-CNN + InfoNCE)...
"%PY%" scripts\train_traj_encoder.py --seq BT-001 --max-frames 2000 --epochs 30 --batch-size 256 --window-size 16 --lr 0.003 --temperature 0.2 --out-dir runs\traj_encoder
if errorlevel 1 goto :error

echo Training finished successfully.
echo - Checkpoint: runs\traj_encoder\traj_encoder.pt
echo - Log: runs\traj_encoder\train_log.csv
popd
exit /b 0

:error
echo Training failed.
popd
exit /b 1
