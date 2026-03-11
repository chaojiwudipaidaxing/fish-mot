@echo off
setlocal
set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%"
python scripts\make_paper_assets.py --main-mean results\main_val\tables\main_table_val_seedmean.csv --main-std results\main_val\tables\main_table_val_seedstd.csv --count-csv results\main_val\tables\count_metrics_val.csv --strong-mean results\main_val\tables\strong_baselines_seedmean.csv --strong-std results\main_val\tables\strong_baselines_seedstd.csv --with-baselines-csv results\main_val\tables\main_table_val_with_baselines.csv --run-config results\main_val\run_config.json --out-dir results\main_val\paper_assets --release-dir results\main_val\release
if errorlevel 1 (
  echo make_paper_assets failed.
  popd
  exit /b 1
)
:: === Degradation Grid (auto) START ===
python scripts\run_degradation_grid.py --run-config results\main_val\run_config.json
if errorlevel 1 (
  echo run_degradation_grid failed.
  popd
  exit /b 1
)
:: === Degradation Grid (auto) END ===
echo Done.
popd
exit /b 0
