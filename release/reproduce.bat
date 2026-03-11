@echo off
setlocal
set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%"
python scripts\make_paper_assets.py --main-mean results\main_table_val_seedmean.csv --main-std results\main_table_val_seedstd.csv --count-csv results\count_metrics_val.csv --strong-mean results\strong_baselines_seedmean.csv --strong-std results\strong_baselines_seedstd.csv --with-baselines-csv results\main_table_val_with_baselines.csv --run-config results\main_val\run_config.json --out-dir results\paper_assets_val --release-dir release
if errorlevel 1 (
  echo make_paper_assets failed.
  popd
  exit /b 1
)
echo Done.
popd
exit /b 0
