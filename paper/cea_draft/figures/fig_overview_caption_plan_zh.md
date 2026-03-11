# CEA 审稿友好图谱（Fig.1–Fig.7）插入建议

以下图已由脚本 `scripts/make_cea_overview_figs.py` 自动生成（PDF+PNG）。

## Fig.1
- 文件：`fig1_framework_architecture.pdf` / `fig1_framework_architecture.png`
- 建议位置：Methods 开头（`section{Methods...}` 后）
- 图注：
  - 中文：面向部署的漂移感知评测框架总体结构。框架由可审计复现核心（C1）、漂移诊断与压力测试层（C2）以及部署决策规则（C3）组成，形成从输入到策略输出的闭环。
  - 英文：Overall deployment-oriented drift-aware evaluation framework, connecting input ingestion, audit-ready reproducibility, stress diagnostics, monitoring loop, and policy output.

## Fig.2
- 文件：`fig2_drift_monitoring_flow.pdf` / `fig2_drift_monitoring_flow.png`
- 建议位置：Drift definition and closed-loop handling 小节（算法前后）
- 图注：
  - 中文：漂移闭环监测与策略切换流程。
  - 英文：Closed-loop drift monitoring and policy switch with threshold-and-persistence triggering.

## Fig.3
- 文件：`fig3_audit_protocol_chain.pdf` / `fig3_audit_protocol_chain.png`
- 建议位置：Audit-ready reproducibility core 小节
- 图注：
  - 中文：审计就绪实验协议链路。
  - 英文：Audit-ready protocol chain using run_config locking and dual SHA256 manifests.

## Fig.4
- 文件：`fig4_deployment_decision_tree.pdf` / `fig4_deployment_decision_tree.png`
- 建议位置：Discussion / deployment guideline 小节
- 图注：
  - 中文：基于资源预算与环境风险的部署决策树。
  - 英文：Deployment decision tree based on runtime budgets and drift-risk context.

## Fig.5
- 文件：`fig5_evaluation_pipeline_overview.pdf` / `fig5_evaluation_pipeline_overview.png`
- 建议位置：Experimental setup 小节（公平评估协议附近）
- 图注：
  - 中文：评测流水线总览。
  - 英文：Evaluation pipeline overview with detector branches, tracker set, and unified evaluator.

## Fig.6
- 文件：`fig6_failure_modes_quad.pdf` / `fig6_failure_modes_quad.png`
- 数据来源：`results/brackishmot_bucket_shift.csv`（用于数值注释）
- 建议位置：Stratified difficulty and failure modes 小节
- 图注：
  - 中文：典型失败模式四宫格（遮挡/密度/转向/低置信度）及对应桶级风险迁移注释。
  - 英文：Representative failure-mode patterns with bucket-level risk deltas (read from BrackishMOT bucket-shift CSV).

## Fig.7
- 文件：`fig7_runtime_stage_breakdown.pdf` / `fig7_runtime_stage_breakdown.png`
- 数据来源：`results/brackishmot/runtime/runtime_profile_e2e_true_summary.csv`
- 建议位置：Runtime and resource profile 小节（Scope B-true 段落）
- 图注：
  - 中文：端到端阶段耗时分解（decode/detector/tracking/write）。
  - 英文：Scope B-true stage-time breakdown (sec/run), showing detector-dominant cost under turbid conditions.

