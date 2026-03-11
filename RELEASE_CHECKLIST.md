# RELEASE_CHECKLIST (Paper Freeze)

## 0) 固定口径环境变量（full）

```bat
set GIT_COMMIT=YOUR_COMMIT_HASH
set MAX_FRAMES=1000
set SEEDS=0,1,2
set DROP_RATE=0.2
set JITTER=0.02
set GATING_THRESH=2000
set METHODS=bytetrack,ocsort,botsort
```

## 1) 先跑主表 + 强 baseline

```bat
scripts\run_main_table_val.bat
scripts\run_strong_baselines_val_multiseed.bat
```

## 2) 跑分桶 / runtime / 论文资产

```bat
.\.venv\Scripts\python.exe scripts\eval_stratified.py --run-config results\main_val\run_config.json --pred-root results\main_val\seed_runs\seed_0
.\.venv\Scripts\python.exe scripts\profile_runtime.py --run-config results\main_val\run_config.json
.\.venv\Scripts\python.exe scripts\make_paper_assets.py --run-config results\main_val\run_config.json
```

## 3) 门禁检查（建议先 preflight，再 paper）

### 3.1 preflight（不等全资产）

```bat
.\.venv\Scripts\python.exe scripts\check_artifacts.py results\main_val\run_config.json --mode preflight --required-gating-thresh 2000
```

预期：
- `Data gate pass`
- `PREFLIGHT CHECKS PASSED.`

### 3.2 paper（全量资产门禁）

```bat
.\.venv\Scripts\python.exe scripts\check_artifacts.py results\main_val\run_config.json --required-gating-thresh 2000 --required-strong-methods ByteTrack,OC-SORT,BoT-SORT
```

预期：
- Data Drift Gate 通过（`seqLength >= max_frames`）
- 主表 + 强 baseline 表齐全且 fresh
- manifest git_commit 非 `unknown`
- 最终 `ALL CHECKS PASSED.`

## 4) 快速定位常见失败

- `manifest git_commit invalid: 'unknown'`
  - 处理：确认先 `set GIT_COMMIT=...`，再重跑 `make_paper_assets.py`
- `strong_baselines_seedmean.csv missing required method rows`
  - 处理：确认 `METHODS=bytetrack,ocsort,botsort` 并重跑强 baseline bat
- `Data gate failed: ... seqLength < ...`
  - 处理：重跑 prepare（full 口径应写入 `data/mft25_mot_full`），不要复用 smoke 的 `mot_root`
- `run-config strict mode blocks legacy/global outputs`
  - 处理：修对应脚本输出路径到 `run_config.result_root` 子目录
