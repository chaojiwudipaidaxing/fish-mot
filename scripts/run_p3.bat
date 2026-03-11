@echo off
setlocal

set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%"

set "DROP_RATE=0.2"
set "JITTER=0.02"

echo [1/19] Prepare MOT data (train_half, 1 sequence, 200 frames)...
python scripts\prepare_mft25.py --splits train_half --seq-limit 1 --max-frames 200 --clean-split
if errorlevel 1 goto :error

if not exist runs\traj_encoder\traj_encoder.pt (
  echo [2/19] traj_encoder.pt not found, train it first...
  python scripts\train_traj_encoder.py --seq BT-001 --max-frames 200 --window-size 16 --epochs 30 --batch-size 32 --lr 0.003 --temperature 0.2 --out-dir runs\traj_encoder
  if errorlevel 1 goto :error
) else (
  echo [2/19] Found runs\traj_encoder\traj_encoder.pt
)

echo [3/19] Run Base(off)...
python scripts\run_baseline_sort.py --split train_half --max-frames 200 --gating off --traj off --alpha 1.0 --beta 0.0 --gamma 0.0 --clean-out --out-dir results\p3\pred_base
if errorlevel 1 goto :error

echo [4/19] Eval Base(off)...
python scripts\eval_trackeval.py --split train_half --tracker-name p3_base --pred-dir results\p3\pred_base --results-csv results\p3\metrics_base.csv --trackers-root results\p3\trackeval_base
if errorlevel 1 goto :error

echo [5/19] Run +gating...
python scripts\run_baseline_sort.py --split train_half --max-frames 200 --gating on --traj off --alpha 1.0 --beta 0.02 --gamma 0.0 --clean-out --out-dir results\p3\pred_gating
if errorlevel 1 goto :error

echo [6/19] Eval +gating...
python scripts\eval_trackeval.py --split train_half --tracker-name p3_gating --pred-dir results\p3\pred_gating --results-csv results\p3\metrics_gating.csv --trackers-root results\p3\trackeval_gating
if errorlevel 1 goto :error

echo [7/19] Run +traj...
python scripts\run_baseline_sort.py --split train_half --max-frames 200 --gating on --traj on --traj-encoder runs\traj_encoder\traj_encoder.pt --alpha 1.0 --beta 0.02 --gamma 0.5 --adaptive-gamma off --clean-out --out-dir results\p3\pred_traj
if errorlevel 1 goto :error

echo [8/19] Eval +traj...
python scripts\eval_trackeval.py --split train_half --tracker-name p3_traj --pred-dir results\p3\pred_traj --results-csv results\p3\metrics_traj.csv --trackers-root results\p3\trackeval_traj
if errorlevel 1 goto :error

echo [9/19] Run +adaptive...
python scripts\run_baseline_sort.py --split train_half --max-frames 200 --gating on --traj on --traj-encoder runs\traj_encoder\traj_encoder.pt --alpha 1.0 --beta 0.02 --gamma 0.5 --adaptive-gamma on --adaptive-gamma-boost 1.5 --clean-out --out-dir results\p3\pred_adaptive
if errorlevel 1 goto :error

echo [10/19] Eval +adaptive...
python scripts\eval_trackeval.py --split train_half --tracker-name p3_adaptive --pred-dir results\p3\pred_adaptive --results-csv results\p3\metrics_adaptive.csv --trackers-root results\p3\trackeval_adaptive
if errorlevel 1 goto :error

echo [11/19] Build table_ablation.csv...
python scripts\build_table_ablation.py --base-csv results\p3\metrics_base.csv --gating-csv results\p3\metrics_gating.csv --traj-csv results\p3\metrics_traj.csv --adaptive-csv results\p3\metrics_adaptive.csv --out-csv results\table_ablation.csv
if errorlevel 1 goto :error

echo [12/19] Run +gating (hard: drop/jitter)...
python scripts\run_baseline_sort.py --split train_half --max-frames 200 --gating on --traj off --alpha 1.0 --beta 0.02 --gamma 0.0 --drop-rate %DROP_RATE% --jitter %JITTER% --clean-out --out-dir results\p3\pred_gating_hard
if errorlevel 1 goto :error

echo [13/19] Eval +gating hard...
python scripts\eval_trackeval.py --split train_half --tracker-name p3_gating_hard --pred-dir results\p3\pred_gating_hard --results-csv results\p3\metrics_gating_hard.csv --trackers-root results\p3\trackeval_gating_hard
if errorlevel 1 goto :error

echo [14/19] Run +traj (hard: drop/jitter)...
python scripts\run_baseline_sort.py --split train_half --max-frames 200 --gating on --traj on --traj-encoder runs\traj_encoder\traj_encoder.pt --alpha 1.0 --beta 0.02 --gamma 0.5 --adaptive-gamma off --drop-rate %DROP_RATE% --jitter %JITTER% --clean-out --out-dir results\p3\pred_traj_hard
if errorlevel 1 goto :error

echo [15/19] Eval +traj hard...
python scripts\eval_trackeval.py --split train_half --tracker-name p3_traj_hard --pred-dir results\p3\pred_traj_hard --results-csv results\p3\metrics_traj_hard.csv --trackers-root results\p3\trackeval_traj_hard
if errorlevel 1 goto :error

echo [16/19] Run +adaptive (hard: drop/jitter)...
python scripts\run_baseline_sort.py --split train_half --max-frames 200 --gating on --traj on --traj-encoder runs\traj_encoder\traj_encoder.pt --alpha 1.0 --beta 0.02 --gamma 0.5 --adaptive-gamma on --adaptive-gamma-boost 1.5 --drop-rate %DROP_RATE% --jitter %JITTER% --clean-out --out-dir results\p3\pred_adaptive_hard
if errorlevel 1 goto :error

echo [17/19] Eval +adaptive hard...
python scripts\eval_trackeval.py --split train_half --tracker-name p3_adaptive_hard --pred-dir results\p3\pred_adaptive_hard --results-csv results\p3\metrics_adaptive_hard.csv --trackers-root results\p3\trackeval_adaptive_hard
if errorlevel 1 goto :error

echo [18/19] Build table_ablation_hard.csv...
python scripts\build_table_ablation.py --base-csv results\p3\metrics_base.csv --gating-csv results\p3\metrics_gating_hard.csv --traj-csv results\p3\metrics_traj_hard.csv --adaptive-csv results\p3\metrics_adaptive_hard.csv --out-csv results\table_ablation_hard.csv
if errorlevel 1 goto :error

echo [19/19] Done.
echo - Table: results\table_ablation.csv
echo - Hard table: results\table_ablation_hard.csv
popd
exit /b 0

:error
echo P3 pipeline failed.
popd
exit /b 1
