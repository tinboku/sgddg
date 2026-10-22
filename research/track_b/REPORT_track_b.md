# Track B 筛选报告：Dataset-Characteristic-Dependent Context

状态：**PLANNED / offline screen only**。没有新 LLM 调用；不把本轮现象当因果结果。

## 结论

29 张可读 subset 表的源 CSV 特征显示了足够的结构差异：行数 5–530,090、列数 1–228、缺失率 0–46.3%、前 100 行 distinct-value 覆盖率 0–1.0%。但现有逐表失败审计只有 C05 的 12 张表，且按“UFD 至少一项事实损失、unsupported 或候选列混淆”定义，12/12 都失败；因此 size/width/missingness 的两半均为 100%，没有可支持的差异证据。schema 启发式几乎全为透明，不能估计该维度。

这不是“没有特征效应”的证明：审计覆盖窄、失败定义宽、且 C05 数据中有通用污染。尤其 datamart 采样范围截断是全体基线问题；两张 trailing-delimiter CSV 错位应排除/单列，不能归因给特征。

## 失败分组

完整数字见 `FEATURES.md` 与 `results/track_b/features.json`。12-table 审计的分组结果：rows low/high = 7/7、5/5；columns low/high = 7/7、5/5；missing low/high = 7/7、5/5。各组均 100%，所以当前数据不能说失败率随这些特征变化。schema 组不可用（12/12 透明，启发式中位数为 1）。

## 真实案例

1. `be2a86f4`：`d_birth_date`/`d_suffix` 发生跨列与 profile/fact 来源混用；源 `data/tables/be2a86f4-03f6-4e89-90b3-cc122c247b74.csv`，缓存 `results/c05_pos/per_table.json`。
2. `1ff67fc2`：`ESTYMD` 的样本 profile 范围压过全表事实；同一 C05 pos 逐表缓存。
3. `3e23114a`：`State` 的 “52 states” 把 `United States` 计入类别；`results/c05/number_audit.json`。

## 候选排序与最小公平对照（均 PLANNED）

| rank | 对比 | 现象/机制假设 | 最小对照与停止条件 |
|---|---|---|---|
| 1 | schema transparent vs opaque | 现有表几乎全透明，无法筛；但别名/编码可能改变 schema grounding 的边际价值。 | 同一值表仅替换可读列名/代码名；相同 100 行、profile、token/输出预算。若事实保留/列混淆无稳定交互，停止。文献支持：Sato 的列语义上下文，不是 DDG 质量；AutoDDG/SGDDG 只提供邻近机制。 |
| 2 | wide vs narrow | 现有宽度 1–228；宽表的 per-column profile 与 prompt overhead 可能挤压有限 UFD。 | 同一语义表按列子集配对，匹配 cells/token budget；报告列级覆盖与缺失，控制行数。若效应仅是 token/长度或 baseline bug，停止。 |
| 3 | low vs high sample coverage / redundancy | 覆盖率从 0–1；固定 100 行可能代表重复大表，也可能漏掉多样值。 | 同域行抽样配对，保持 rows/token 与 truth 不变，只改变 diversity/coverage；若描述差异不超过复跑噪声，停止。 |

行数仍是锚点，但不列为已支持 top-3：当前 12-table 失败率两侧饱和，无法区分 size effect。missingness、long-tail、dependency、temporal shift、mixed types 仍是 **UNTESTED/DATA GAP**，不宣称负结果。

## 文献边界

AutoDDG 可支持固定采样、profile/context 与 UFD/SFD 的任务边界；不能支持“按数据特征条件化失败”的已证结论。Less Is More?（arXiv:2606.02334）支持输入上下文比较的邻近动机，不支持 dataset-characteristic policy。TAP4LLM（2024）支持预算化采样/packing 在表推理中的设计启发，不支持 DDG 质量因果迁移。两者均不能替代本任务的真实描述审计。

## NO-GO 门

对现有审计数据的 Track B 因果筛选：**NO-GO（暂不批准付费 pilot）**。原因是分组失败率完全饱和，且污染/覆盖限制未解除。若研究者要继续，先批准最小 paired schema/width/coverage 之一，并冻结去污染表集、同预算控制、重复噪声基线和人工事实审计。
