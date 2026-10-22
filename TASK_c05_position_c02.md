# 任务：C05 位置对齐对照 + C02 问题验证 + 同步 GitHub

你是实现负责人。当前状态：C05 筛选"有保留通过"（归因不纯：表示形式/事实位置/指针措辞同时变了）；Codex 建议位置对齐对照。代码在分支 c05-structure-input（c05_structured.py、c05_run.py、REPORT_c05.md）。协议文件在 /Users/mo/.hermes/cache/scratch/hcresearch/（AGENTS.md、research/CANDIDATES.md C02 节）。

## 第一部分：C05 位置对齐对照（API ≤$0.05，gpt-4o-mini T=0 走缓存）
- 固定各槽位的位置和指令措辞（用原版臂的槽位布局），**只改变 profile 内容的表示形式**（自然语言 vs 键值分组）——即两臂 prompt 结构逐字节同构，只有 profile 槽内文本不同。
- Codex 特别指出：只改 facts 槽不够，要隔离整个 profile 处理。
- 12 张冻结表（configs/c05_tables.txt），跑 UFD+SFD，记录与之前同口径的指标（事实保留、列混淆、token）。
- 报告结果追加到 REPORT_c05.md 新章节"位置对齐对照"，并修正 Codex 指出的三处报告问题（12/12 口径过宽、925 数字、跨列拼接例）。

## 第二部分：C02 问题验证（零 API，纯本地）
- C02 判据前置："如果真实数据中很少出现联合覆盖问题，则降低优先级"——先验证问题是否存在。
- 本地检查 30 张表：每张表有时间和地区/类别列的，用 groupby 算实际联合覆盖（如 A 地区只有 2010-2014），对照 bugfix 的 UFD/SFD 文本，找"描述声称的覆盖"与"实际联合覆盖"不一致的案例。
- 输出：有多少表存在真实联合覆盖误述、给 2-3 个具体案例原文。结果写入 REPORT_c02_prevalence.md。

## 第三部分：同步 GitHub
- commit 两仓库改动（分支上）
- 把 REPORT_c05.md 更新版、REPORT_c02_prevalence.md、REPORT_s3_sampling.md 复制到 /Users/mo/.hermes/cache/scratch/hcresearch/research/，commit + push
- 在 sgddg 的 Draft PR #1 加评论：位置对齐结果 + C02 问题验证结果

## 硬约束
- API ≤$0.1（本任务合计）；不改动 AutoDDG 原包；不跑冻结清单外的表；不输出 key；正负如实报
- 最终回复只给要点
