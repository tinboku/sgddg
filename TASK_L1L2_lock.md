# 任务：实现"程序化锁定 UFD→SFD 信息保留"（①号提议）+ 30 表对照

你是实现负责人（本次由 Codex 担任，Claude 额度恢复前由你顶上）。背景与已完成的工作见 ROADMAP_20261021.md（先读）。你已诊断出最大损失："事实进 UFD 后，SFD 拿 UFD 重新生成时丢了信息"（−0.012）。本任务实现程序化保留机制并跑小规模对照。

## 实现内容（external/AutoDDG 内，不改原包）
1. **锁定版 SFD 生成**（新文件或子类，如 description/search_locked.py）：
   - SFD 的 Dataset Overview 段**由程序从 UFD 原样复制**，LLM 只生成主题/关键词/应用补充段
   - 输出预算与现有 F2 臂相同（等长对照）
2. **事实覆盖检查**：
   - 已选事实（F2 缓存里那份带 ID 的选择清单）逐条检查在最终 SFD 文本中的覆盖情况（容忍 "2023-Q1"→"2023 Q1" 写法归一，复用 05_fact_audit.py 已有的归一逻辑）
   - 遗漏的事实做**定点修补**：把遗漏事实文本追加到对应段落（程序追加，不走 LLM 重写）；或最多一次 LLM 修补调用，失败则程序追加兜底
3. **两个新臂配置**（experiments/autoddg_deep/configs/）：
   - **L1** = 锁定复制（只锁 overview，无覆盖检查）
   - **L2** = 锁定复制 + 事实覆盖检查 + 定点修补
   - 与 F2 共享 profile 缓存和事实选择（L1/L2 只改 SFD 生成这一步，严格单变量）

## 实验（subset30 那 30 张表，不许跑更多）
- 跑 L1 r1、L2 r1（OPENAI_API_KEY 在 ~/.zshrc，先 source；L2 的修补调用也走缓存）
- 评测：03_index_eval.py（BM25，设定与之前一致：sfd/per_query）+ 04_paired_test.py
- 主比较：**L2 vs bugfix**（锁定+覆盖检查的总效果）、**L2 vs F2**（锁定 vs 重新生成——这个对照直接量化"再生成丢信息"值多少分）、L1 vs F2（纯锁定）、L1 vs L2（覆盖检查的增量）
- 所有数字放噪声底（baseline r1 vs r2）旁边解读
- 05_fact_audit.py 统计 L1/L2 的事实保留率（对比 F2 的 99.6%——L2 应接近 100%）

## 报告与收尾
- 中文简报 experiments/autoddg_deep/REPORT_L1L2.md：实现内容、四组对照（L2 vs bugfix / L2 vs F2 / L1 vs F2 / L1 vs L2）的逐查询结果+p 值+噪声底、保留率对比、结论（锁定机制能不能阻止损失、覆盖检查值多少）
- 所有新改动 commit（两个仓库分开说清楚）
- 结果正负都如实报

## 硬约束
- 不改动 AutoDDG 原包文件；不跑 30 张之外的表
- 不输出 key 值
- 工作区里 Claude 留下的缓存和 commit 是进度接缝，接着用，不要重做
