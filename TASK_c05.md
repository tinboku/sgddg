# 任务：C05 Structure-Preserving Context Representation 最小 paired diagnostic

你是实现负责人。先读三份文件：/Users/mo/.hermes/cache/scratch/hcresearch/HERMES_START_HERE.md（当前最高优先级指令，严格照办）、同目录 research/ROADMAP_CURRENT.md、research/CANDIDATES.md 的 C05 节。新协议关键点：这是**筛选阶段的质量对照实验**，不是检索实验；结果经 Draft PR 供 ChatGPT 审查。

## 研究问题（照 CANDIDATES C05）
Profile 序列化会改变 LLM 把事实关联到列的方式，并影响输入 token。**假设**：紧凑结构化表示在同等信息量下保留列-事实关系，质量/成本更优。

## 最小 paired diagnostic（严格照 HERMES_START_HERE 第 3 条）
1. **工作分支**：在 sgddg-github 新建分支 c05-structure-input（不直接动 main）。
2. **冻结表清单**：从 subset30 里选 8-12 张可运行表（含典型列和复杂列：混合格式、长尾类别、哨兵值），存 configs/c05_tables.txt 并冻结。**不得用 NTCIR 测试查询标签指导选表。**
3. **两种表示，完全同一份 Profile 和事实**（复用 F2 缓存里已有的 profile/事实，不重新提取）：
   - **原版**：AutoDDG 原始自然语言 profile 输入
   - **结构化**：一种清晰紧凑表示——建议"键值分组"：每列一个块（name: X; type: Y; range: [a,b]; top_values: ...; missing: z%），表级事实一组（temporal/spatial/granularity）；共享值用字典引用。不引入图像/图结构。
4. **生成**：两种表示各跑 UFD+SFD（gpt-4o-mini，T=0，走缓存——相同输入命中缓存；只有输入不同的调用是新花费，预估 ≤$1）。其余参数（prompt 模板、解码、预算）两臂完全一致，只换 profile 输入的表示形式。
5. **记录**（每表）：
   - 逐列事实对应：每列的统计在两种表示里是否都出现（覆盖率）
   - UFD/SFD 文本；描述里的事实与 profile 的对应（忠实性：事实逐条核对）
   - **列混淆检查**：描述里有没有把 A 列的统计说成 B 列的（配对原 profile 逐条核对）
   - 全流程 token（输入/输出分开）和费用
6. **质量评价按 C05 判据**：列混淆率、忠实性、关键信息保留、token——**不做 NDCG**。

## 交付
1. 中文报告 experiments/autoddg_deep/REPORT_c05.md：假设、两臂设计、逐表结果、列混淆率、忠实性、token/费用、失败案例、按 C05 停止判据的结论（"只是更短没有质量效应"就如实说）
2. 在分支上 commit，推送到 origin（tinboku/sgddg）
3. 用 gh 创建 **Draft PR**（到 tinboku/sgddg 的 main）：按 hcresearch 仓库 PR #1 里的 .github/pull_request_template.md 模板填写（hypothesis/prior work/changes/comparison/results/review request）。**不 merge。**
4. 最终回复只给要点（结论、token、费用、PR 链接）

## 硬约束
- 不改动 AutoDDG 原包既有文件；新代码走子类/新文件
- 只跑冻结的 8-12 张表；API ≤$1（gpt-4o-mini，缓存优先）
- 不用测试查询标签选表；结果正负如实报
- 允许读写 experiments/autoddg_deep/ 全部内容和两仓库 git 操作
- 已有缓存/commit 是接缝，不要重做
