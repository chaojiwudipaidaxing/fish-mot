# Executive Summary

本次审计基于仓库内**实际代码与产物**完成（未做全量重跑），结论如下：

| 结论项 | Y/N | 结论 |
|---|---|---|
| Data Drift Gate (Y/N) | **N** | `run_config` missing `mot_root`, and current `data/mft25_mot/val_half` has `seqLength=1` for all sequences; gate failed. |
| 工程与复现审计是否达“可投稿定稿”标准 | **N** | 产物口径发生 smoke/full 混用；`run_config` 与最终表格路径不一致；`manifest.git_commit=unknown` 触发 fail-fast。 |
| 你方方法 vs 强 baseline 是否可横向对比 | **N（当前主表）** | 当前主表/合并表是 smoke 口径（`max_frames=1`，seed=0），无法代表正式对比；历史 full 结果存在但未被当前权威表引用。 |
| 论文“逐步提升”主张是否被当前数值支持 | **N** | 在可见 full 产物（seed1/2）里，`+gating/+traj/+adaptive`均低于 Base，且明显落后 ByteTrack。 |
| 现在是否适合投 CEA | **N（可快速补齐后转 Y）** | 选题与 CEA scope 高匹配，但实验口径与复现证据未闭环，当前稿件风险高。 |
| 当前是否具备“冲 Q1”条件 | **N** | 缺统一正式口径全量结果、统计显著性闭环、强 baseline 稳定对比与可复现包完整性。 |

**三条最关键结论：**
1. 目前“权威输出”被 smoke 结果污染（`max_frames=1`），导致主表与结论不可用于投稿定稿。  
2. 代码层面的防漂移机制已具雏形（`run_config` 路由、`check_artifacts.py`、`manifest` 哈希），但正式口径尚未完成一次闭环执行。  
3. 从已有 full 历史结果看，论文叙事必须从“绝对提升”转向“受控退化下的鲁棒性/可解释权衡”，否则不被数据支持。

---

# Reproducibility & Drift Audit

## 1) 关键配置与目录路由检查

### 1.1 当前 `main_val` run_config（被当作正式入口）
- 文件：`results/main_val/run_config.json`
- 关键字段：
  - `split=val_half`
  - `max_frames=1`
  - `seeds=[0]`
  - `drop_rate=0.2`
  - `jitter=0.02`
  - `gating_thresh=9.210340371976184`
  - `git_commit=unknown`
- 问题：该文件是 smoke 口径，且为旧 schema（缺 `result_root/tables_dir/paper_assets_dir/release_dir/runtime_dir/stratified_dir`）。

### 1.2 smoke run_config（新 schema）
- 文件：`results/_smoke/main_val/run_config.json`
- 含完整路径映射：`result_root/tables_dir/paper_assets_dir/release_dir/runtime_dir/stratified_dir`。
- 但仍是 smoke：`max_frames=1`、`seeds=[0]`。

## 2) 产物是否写入 run_config 指定结果根

### 2.1 `results/main_val` 下缺失新布局产物（FAIL）
以下文件在 `results/main_val/...` 下均缺失：
- `results/main_val/tables/main_table_val_seedmean.csv`
- `results/main_val/tables/strong_baselines_seedmean.csv`
- `results/main_val/tables/main_table_val_with_baselines.csv`
- `results/main_val/runtime/runtime_profile.csv`
- `results/main_val/stratified/stratified_metrics_val.csv`
- `results/main_val/paper_assets/paper_main_table.tex`
- `results/main_val/release/manifest.json`

### 2.2 反而写到了 legacy/global 或 `_smoke`（DRIFT）
- legacy/global：`results/main_table_val_seedmean.csv`、`results/main_table_val_with_baselines.csv`、`results/runtime_profile.csv` 等
- smoke 根：`results/_smoke/main_val/tables/*`、`results/_smoke/main_val/paper_assets/*`、`results/_smoke/main_val/release/*`

## 3) freshness 证据

- `results/main_table_val_with_baselines.csv` mtime 为 `2026-03-01 23:08:58`，不早于其输入：
  - `results/main_table_val_seedmean.csv` (`23:08:44`)
  - `results/strong_baselines_seedmean.csv` (`23:08:58`)
- `_smoke` 目录下同样 freshness 成立。

> 结论：**合并表自动重建/新鲜度机制本身有效**，但“输入来源已经是 smoke 口径”导致结果仍不可用于正式论文。

## 4) fail-fast 检查结果

执行：
- `python scripts/check_artifacts.py results/main_val/run_config.json`  
  - 失败：`max_frames=1 < 1000`
- `python scripts/check_artifacts.py results/_smoke/main_val/run_config.json --required-max-frames 1 --required-seeds 0`  
  - 失败：`manifest git_commit invalid: 'unknown'`

> `git_commit=unknown` 导致失败是**设计使然（正确）**。通过方式：
- 在真实 git 环境运行（`git rev-parse HEAD` 可得 commit），或
- 运行前注入 `GIT_COMMIT` 环境变量。

## 5) manifest / reproduce 检查

- 缺失：`release/manifest.json`（global 根目录不存在）
- 存在：`release/reproduce.bat`，但仍引用 legacy 全局路径（非 `result_root` 路由）。
- `_smoke` 下 `manifest.json` 存在，含 SHA256 与 run_config，但 `git_commit=unknown`。

## 6) 同名资产多目录并存（易误引用）

- 存在 `results/paper_assets/` 与 `results/paper_assets_val/`，另有 `results/_smoke/main_val/paper_assets/`。
- 同名图表在不同目录并存，且时间戳不同，极易误引用。

---



# Data Drift Gate (P0 mandatory gate)

Gate rule: if any Data Drift sub-gate fails, this audit must stay "Not submission-ready (N)".

## A) mot_root integrity and frame-count consistency gate

### A.1 Is `mot_root` declared in run_config?
- Checked: `results/main_val/run_config.json`
- Result: **FAIL** (`mot_root` field missing)
- Blocking impact: we cannot prove which MOT data root was used by this run.

### A.2 Per-sequence `seqLength` check
- Checked root: `data/mft25_mot/val_half`
- Evidence (path + seqLength + mtime):
  - `data/mft25_mot/val_half/BT-001/seqinfo.ini` -> `seqLength=1` (mtime `2026-03-02 01:37:23`)
  - `data/mft25_mot/val_half/BT-003/seqinfo.ini` -> `seqLength=1` (mtime `2026-03-02 01:37:23`)
  - `data/mft25_mot/val_half/BT-005/seqinfo.ini` -> `seqLength=1` (mtime `2026-03-02 01:37:23`)
  - `data/mft25_mot/val_half/MSK-002/seqinfo.ini` -> `seqLength=1` (mtime `2026-03-02 01:37:23`)
  - `data/mft25_mot/val_half/PF-001/seqinfo.ini` -> `seqLength=1` (mtime `2026-03-02 01:37:23`)
  - `data/mft25_mot/val_half/SN-001/seqinfo.ini` -> `seqLength=1` (mtime `2026-03-02 01:37:23`)
  - `data/mft25_mot/val_half/SN-013/seqinfo.ini` -> `seqLength=1` (mtime `2026-03-02 01:37:23`)
  - `data/mft25_mot/val_half/SN-015/seqinfo.ini` -> `seqLength=1` (mtime `2026-03-02 01:37:23`)
- Result: **FAIL** (for paper full-run target `max_frames=1000`, this data root is truncated to single frame).

### A.3 Why this invalidates ID metrics
- With single-frame sequences, cross-frame identity association is mostly absent.
- `IDSW` can stay near 0 and `IDF1/AssA/HOTA` lose tracking-identity interpretability.

## B) GT identity quality file alignment with current run_config/data

### B.1 mtime alignment check
- `results/gt_id_quality_val_half.csv` mtime: `2026-03-01 19:14:58`
- `results/main_val/run_config.json` mtime: `2026-03-01 23:08:31`
- `data/mft25_mot/val_half/BT-001/seqinfo.ini` mtime: `2026-03-02 01:37:23`
- Result: **FAIL** (`gt_id_quality_val_half.csv` is older than current run_config and older than rebuilt MOT data).

### B.2 Additional evidence
- Recomputed file `results/gt_id_quality_val_half_current.csv` (mtime `2026-03-02 02:22:03`) shows 8/8 FAIL (median_len=1, pct_len1=1.0), matching current single-frame data.
- This proves the older PASS-like file is stale for current data.

### B.3 Minimal fix
- When generating GT quality CSV, write metadata: `run_id`, `git_commit`, `max_frames`, `mot_root`.
- Track this file/path/hash in manifest for audit alignment.

## C) Prepare overwrite risk gate (must prevent smoke overriding full)

### C.1 Static check
- `scripts/prepare_mft25.py` supports `--out-root` (good), but default is `data/mft25_mot`.
- `scripts/run_main_table_val.bat:43` calls prepare without `--out-root`:
  - `python scripts\prepare_mft25.py --splits val_half --max-frames %MAX_FRAMES% --clean-split`
- `scripts/run_baseline_sort.py` default `--mot-root` is `data/mft25_mot`, and `run_main_table_val.bat:60/65/70/75` does not pass `--mot-root`.
- `scripts/write_run_config.py` does not include `mot_root`; `scripts/run_strong_baselines.py` apply_run_config also does not override `mot_root` from run_config.
- Result: **FAIL (High Risk)**.

### C.2 Dynamic contradiction evidence
- Historical full log: `results/run_val_drop02.log:23` reports `frames=6254`.
- Current MOT root: all `seqLength=1` in `data/mft25_mot/val_half/*/seqinfo.ini`.
- Conclusion: same data root was reused and overwritten, causing drift.

## Data Drift Gate summary table

| Gate | Pass/Fail | Evidence | Fix |
|---|---|---|---|
| seqLength >= max_frames | **FAIL** | All `data/mft25_mot/val_half/*/seqinfo.ini` show `seqLength=1`; full-paper target is `max_frames=1000`. | Use isolated MOT roots: `data/mft25_mot_full` and `data/mft25_mot_smoke` (or `<result_root>/mot`). |
| GT quality aligned with run_config | **FAIL** | `results/gt_id_quality_val_half.csv` is older than current run_config and current rebuilt seqinfo files. | Add `run_id/git_commit/max_frames/mot_root` metadata and manifest hash linkage. |
| prepare does not overwrite full | **FAIL** | prepare called without `--out-root`; baseline/eval defaults still point to `data/mft25_mot`; full vs current seqLength contradiction exists. | Add `mot_root` to run_config; force all prepare/tracker/eval scripts to read it; add data gate in `check_artifacts.py`. |

## Mandatory minimal patch set

1. Add `mot_root` into `scripts/write_run_config.py`:
   - smoke default: `data/mft25_mot_smoke`
   - full default: `data/mft25_mot_full`
2. In `run_main_table_val.bat` and `run_strong_baselines_val_multiseed.bat`:
   - prepare must pass `--out-root <mot_root>`
   - tracker/eval must pass `--mot-root <mot_root>`
3. In `scripts/check_artifacts.py` add data gate:
   - fail if sequence `seqLength` is below required official frame cap.
4. Under `--run-config`, enable strict no-legacy-write.

Final gate decision: **Data Drift Gate = FAIL (P0 Blocking)**. Therefore the overall audit decision remains **N**.


# Benchmark Fairness & Config Consistency

## 1) 当前“主表/合并表”并非正式多 seed 全量

- `results/main_table_val_seedmean.csv`：4 行（Base/+gating/+traj/+adaptive），全部同值，`IDSW=0`。
- `results/main_table_val_seedstd.csv`：全 0。
- `results/per_seq_main_val.csv`：仅 `seed=0`，且 `used_frames` 全为 1。

**判定：当前主表是 smoke 结果，不满足“多 seed + max_frames=1000”的正式对比要求。**

## 2) 强 baseline 多 seed 一致性

- `results/strong_baselines_seedmean.csv` 仅有 `ByteTrack/OC-SORT` 两行，`BoT-SORT` 缺失。
- `results/strong_baselines_seedstd.csv` 全 0，表明当前聚合来自单 seed 或等价单次结果。

## 3) 代码公平性（静态）

- `scripts/run_strong_baselines.py` 中三方法共用 `run_baseline_sort.py` + 同一 `--det-source` / `drop-rate` / `jitter` 参数流，设计上可实现公平输入。
- 但当前产物层面未形成单一正式 run_config 口径闭环，因此**公平性在“代码上成立、在当前表格上不成立”。**

## 4) 数据准备状态冲突（关键）

- 当前 `data/mft25_mot/val_half/*/seqinfo.ini` 的 `seqLength` 全为 1（总帧 8）。
- 但历史 per-seq 结果（如 `results/main_val/seed_runs/seed_1/per_seq_base.csv`）显示 `used_frames` 达 1000/916/876...（总计 6254）。

**判定：同一路径下存在“历史 full + 当前 smoke”混存，造成口径漂移。**

---

# Results Validity & Claim Fit

## 1) 指标量纲 sanity

- `scripts/eval_trackeval_per_seq.py` 里：`HOTA/DetA/AssA/IDF1` 统一乘 100；`IDSW` 取 CLEAR 原始计数（不乘 100）。
- 量纲逻辑正确；但下游表格混用 smoke/full 会掩盖真实趋势。

## 2) IDSW 长期为 0 的定位

### 2.1 当前权威表中 IDSW=0 的直接原因
- `results/per_seq_main_val.csv` 当前仅 `used_frames=1`（每序列单帧），自然几乎不可能发生 ID switch。
- 对应 `id_eval_valid=0` 与 `degenerate_gt_ids(single_frame_tracks)` 可在 `results/main_val/seed_runs/seed_0/per_seq_base.csv` 看到。

### 2.2 GT 质量文件的“表面 PASS”是陈旧结果
- `results/gt_id_quality_val_half.csv` 显示 PASS（median_len 大于 1），但时间早于当前 smoke 数据重制。
- 重新跑得到：`results/gt_id_quality_val_half_current.csv`，8/8 序列 FAIL（median_len=1, pct_len1=1）。

> 结论：IDSW=0 不是评测代码量纲错误，而是**当前输入变成单帧 smoke**导致 ID 指标失去解释性。

## 3) 当前数值是否支持“逐步提升”

### 3.1 当前“权威合并表”（smoke）
- Base/+gating/+traj/+adaptive/ByteTrack 全部 `HOTA=91.957, IDF1=93.407, IDSW=0`。
- 该表不具判别力，不能用于论文结论。

### 3.2 历史 full 产物（seed1/2，仅作诊断，不可直接当最终表）

你方方法（seed1/2 平均）：
- Base: HOTA **42.880**, IDF1 **52.546**, IDSW **91.125**
- +gating: HOTA **26.832**, IDF1 **27.624**, IDSW **238.750**
- +traj: HOTA **26.995**, IDF1 **27.861**, IDSW **237.250**
- +adaptive: HOTA **26.852**, IDF1 **27.427**, IDSW **238.875**

强 baseline（seed1/2 平均）：
- ByteTrack: HOTA **58.626**, IDF1 **64.393**, IDSW **88.438**
- OC-SORT: HOTA **33.840**, IDF1 **29.514**, IDSW **333.062**
- BoT-SORT: HOTA **35.168**, IDF1 **30.836**, IDSW **326.625**

**判定：当前不支持“Base→+gating→+traj→+adaptive持续提升”叙事。**

## 4) 可写叙事建议（基于现有证据）

禁止写法：
- “我们方法整体优于强 baseline / 持续提升主指标”

可写替代：
- “在受控检测退化（drop/jitter）与资源约束下，提出可复现实验管线，展示不同关联策略的 accuracy-resource trade-off，并分析失效模式（IDSW 上升机制）。”

## 5) `+adaptive` 断崖解释性

- 代码已有 clamp 参数（`adaptive_gamma_min/max`）和 `--frame-stats` 开关。
- 但当前存档日志多数 `frame_stats=off`，缺乏逐帧 match/lost/new 的证据链。
- 因此审稿时难解释为何 +adaptive 大幅退化（在历史 full 结果里明显存在）。

---

# Venue Fit: CEA

## 1) 联网证据（含日期）

> 访问日期：2026-03-02

1. Elsevier / ScienceDirect CEA insights：
   - 链接：<https://www.sciencedirect.com/journal/computers-and-electronics-in-agriculture/about/insights>
   - 可见信息：影响力指标、接受率（页面显示约 16%）、年度发表量等。
2. Elsevier CEA Guide for Authors（Aims & Scope）：
   - 链接：<https://www.sciencedirect.com/journal/computers-and-electronics-in-agriculture/about/guide-for-authors>
   - 可见信息：强调电子/信息技术在农业场景应用与工程系统研究。
3. Wageningen University library journal page（汇总 JCR/Scopus/SJR信息）：
   - 链接：<https://library.wur.nl/WebQuery/journals/177460>
   - 可见信息：Scopus/CiteScore/SJR、分类与分区等（部分 JCR 信息需要机构权限）。

## 2) 主题匹配度判断

- 你的项目（鱼类/养殖多目标跟踪、计数稳定性、部署资源画像、可复现实验工程）与 CEA 的应用导向高度匹配。
- **匹配度评分：8.5/10**（方向匹配高；当前短板主要在实验闭环与结论强度，而非选题）。

## 3) 现在能否投 CEA

- **当前：N**（因为结果口径未闭环，不是选题不匹配）。
- 若完成一次正式闭环（统一 run_config + 全量多 seed + artifact guard pass），可转为 **Y（可投）**。

---

# Q1 Feasibility

## 1) 分区信息与证据（联网）

- CEA 在公开可见的 Scopus/SJR 聚合中通常属于 Q1 区间（农业信息化相关类别）。
- JCR 最新完整分区需 Clarivate JCR 权限核验；当前仓库环境无法直接访问 JCR 原始库页面。
- 可用替代证据：WUR 页面与 Elsevier insights 的高影响指标，支持“Q1 级别竞争强度高”的判断。

> 审计意见：Q1 目标本身合理，但**当前证据链不足以支撑“现在就冲 Q1”。**

## 2) 审稿人视角：冲 Q1 最低条件（必须满足）

1. **强 baseline 竞争性**
   - 至少在某些受控设定（如 degrade/high-density/occlusion bucket）有统计显著优势，或给出可解释 trade-off 优势。
2. **消融完整性**
   - Base/+gating/+traj/+adaptive 的每一步贡献可重复、可解释，且不依赖单次 seed。
3. **规模与泛化**
   - 不仅单 split 单配置；至少多 seed + 分桶 + 场景差异讨论。
4. **统计显著性**
   - mean±std + 置信区间/显著性检验（至少对核心主张）。
5. **应用价值闭环**
   - 计数稳定性、资源画像（FPS/CPU/Mem）、部署约束下的推荐策略。

## 3) 当前是否具备冲 Q1条件

- **N**

## 4) 最短差距路线图（可执行）

1. 统一正式口径重跑一次（禁止 smoke 混入）：
   - 目标：`max_frames=1000`, `seeds=0,1,2`, `drop=0.2`, `jitter=0.02`, 固定 `GATING_THRESH`。
2. 生成并通过 artifact gate：
   - `python scripts/check_artifacts.py results/main_val/run_config.json`
   - 必须通过 `max_frames`、freshness、manifest commit 校验。
3. 重建论文资产（仅从 `result_root`）：
   - 生成 main/count/with_baselines/stratified/runtime 的表图 tex/png。
4. 若核心方法仍不优于 ByteTrack：
   - 立即切换主张到“鲁棒性+资源权衡+可复现工程”，并用 stratified/count/runtime 支撑。

---

# Blocking / Non-blocking Issues

| 优先级 | 类型 | 问题 | 证据 | 风险 | 修复建议 |
|---|---|---|---|---|---|
| P0 | **Blocking** | Data Drift Gate fails (`mot_root` missing and MOT data truncated) | `results/main_val/run_config.json`, `data/mft25_mot/val_half/*/seqinfo.ini`, `results/run_val_drop02.log:23` | Full-paper metrics become non-interpretable and can look artificially high | Add `mot_root` in run_config, isolate full/smoke MOT roots, and add seqLength gate in `check_artifacts.py` |
| P0 | **Blocking** | 正式 run_config 为 smoke（`max_frames=1`） | `results/main_val/run_config.json` | 所有“主表”不可投稿使用 | 用正式参数重写 run_config 并全流程重跑 |
| P0 | **Blocking** | `results/main_val/...` 新布局产物缺失 | `results/main_val/tables` 等目录不存在 | “单一口径源”名存实亡 | 所有脚本统一 `--run-config`，仅写 `result_root` |
| P0 | **Blocking** | `manifest.git_commit=unknown` | `results/_smoke/main_val/release/manifest.json` | 无法通过可复现审计 | 在 git 仓库运行或设置 `GIT_COMMIT` |
| P0 | **Blocking** | 主表/强 baseline 当前是 smoke 且单 seed | `results/main_table_val_seedmean.csv`, `results/strong_baselines_seedstd.csv` | 横向比较失真 | 重新跑多 seed 全量，输出到同一 `result_root` |
| P0 | **Blocking** | GT ID 质量当前退化（单帧） | `results/gt_id_quality_val_half_current.csv` 全 FAIL | IDF1/IDSW 无解释性 | 先 `prepare --max-frames 1000` 再重评 |
| P1 | **Blocking** | 分桶结果仍出现 `quantile_search_unmet` / two-bin fallback | `results/stratified_metrics_val.csv` note 列 | 论文分桶严谨性被质疑 | 完成稳定三桶策略并在图表说明合并逻辑 |
| P1 | Non-blocking | `paper_assets` / `paper_assets_val` / `_smoke/.../paper_assets` 并存 | 多目录同名文件 | 易误贴图、误贴表 | 论文只认 `result_root/paper_assets`，其余归档 |
| P1 | Non-blocking | runtime 仅 smoke 规模 | `results/runtime_profile.csv` (`max_frames=20`) | 资源结论不可外推 | 跑 full runtime 并输出同口径图表 |
| P2 | Non-blocking | `+adaptive` 退化缺逐帧证据 | 日志多数 `frame_stats=off` | 机理论证薄弱 | 开 `frame_stats=on` 采样日志用于 case study |

---

# Action Plan

## Priority 0: 先拿到“可投稿口径”一套完整产物

1. **重建正式数据与主表（禁止 smoke）**
```bat
set GATING_THRESH=2000
set MAX_FRAMES=1000
set SEEDS=0,1,2
set DROP_RATE=0.2
set JITTER=0.02
set GIT_COMMIT=<your_real_commit_hash>
scripts\run_main_table_val.bat
```

2. **重跑强 baseline 多 seed（同口径）**
```bat
set MAX_FRAMES=1000
set SEEDS=0,1,2
set PREPARE=no
set METHODS=bytetrack,ocsort,botsort
scripts\run_strong_baselines_val_multiseed.bat
```

3. **补 stratified / runtime / paper assets（同 run_config）**
```bat
python scripts\eval_stratified.py --run-config results\main_val\run_config.json --pred-root results\main_val\seed_runs\seed_0
python scripts\profile_runtime.py --run-config results\main_val\run_config.json
python scripts\make_paper_assets.py --run-config results\main_val\run_config.json
```

4. **强制审计门禁**
```bat
python scripts\check_artifacts.py results\main_val\run_config.json
```

## Priority 1: 结论与叙事修正

5. 若 full 多 seed 后你方仍不超 ByteTrack：
- 将主张改为：**受控退化下鲁棒性与资源权衡**（非 SOTA 提升）。
- 主表 + 分桶 + 计数稳定 + runtime 四证据联动。

6. 若要继续冲“方法提升”：
- 优先定位 `+adaptive` 退化原因（代价归一化、门控尺度、生命周期参数），并提供逐帧匹配统计证据。

## Priority 2: 文稿工程与复现包收口

7. 统一论文资产目录，仅使用：`results/main_val/paper_assets/`。
8. 在文中与补充材料给出：`run_config.json` + `manifest.json` + `reproduce.bat`。

---

## 审计结论（最终）

- **现在是否可以进入“写论文定稿阶段”：N**  
- **最主要阻塞项：**正式口径未闭环（smoke 污染 + run_config 路径漂移 + manifest commit 未锁定）。
- **可修复性：高**（工程骨架已具备，按 Action Plan 1 轮即可补齐）。

