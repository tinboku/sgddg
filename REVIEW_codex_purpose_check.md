我会先核对 AutoDDG 原文中与目标、生成流程和评测有关的部分，再给出路线取舍；全程只读，不改文件。
web search: 
web search: https://arxiv.org/abs/2502.01050
exec
/bin/zsh -lc "rg --files -g '*SGDDG*' -g '*sgddg*' -g 'AGENTS.md' -g '*README*' -g '*autoddg*' -g '*AutoDDG*'" in /Users/mo/Desktop/sgddg-github
 succeeded in 0ms:
REVIEW_claude_autoddg_deep.md
PLAN_autoddg_deep_v1.1.md
REVIEW_codex_autoddg_deep.md
README.md
PLAN_REVIEW_autoddg_deep.md
experiments/README.md
experiments/agent_metadata_gen/pilot/sgddg_bridge.py
experiments/agent_metadata_gen/pilot/README.md
experiments/relation_aware_pipeline/paper_artifacts/figures/README.md
experiments/autoddg_deep/README.md
experiments/autoddg_deep/constraints_autoddg_3c59e1c.txt
experiments/business_rule_claims/README_zh.md
experiments/relation_aware_pipeline/out_astro2/autoddg_3.jsonl
experiments/relation_aware_pipeline/out_astro2/autoddg_eval.log
experiments/relation_aware_pipeline/out_astro2/autoddg_1.jsonl
experiments/relation_aware_pipeline/out_astro2/autoddg_5.jsonl
experiments/relation_aware_pipeline/out_astro2/autoddg_0.jsonl
experiments/relation_aware_pipeline/out_astro2/autoddg_2.jsonl
experiments/relation_aware_pipeline/out_astro2/autoddg_pointwise.csv
experiments/relation_aware_pipeline/out_astro2/autoddg_protocol.json
experiments/relation_aware_pipeline/out_astro2/autoddg_4.jsonl
experiments/relation_aware_pipeline/out_astro2/autoddg_struct_3.jsonl
experiments/relation_aware_pipeline/out_astro2/autoddg_struct_1.jsonl
experiments/relation_aware_pipeline/out_astro2/autoddg_struct_0.jsonl
experiments/relation_aware_pipeline/out_astro2/autoddg_pairwise.csv
experiments/relation_aware_pipeline/out_astro2/autoddg_struct_2.jsonl
experiments/relation_aware_pipeline/evaluate_autoddg_protocol.py
experiments/relation_aware_pipeline/ramp/autoddg_gen.py
experiments/business_rule_claims/README.md
experiments/relation_aware_pipeline/out_santos/autoddg_3.jsonl
experiments/relation_aware_pipeline/out_santos/autoddg_1.jsonl
experiments/relation_aware_pipeline/out_santos/autoddg_5.jsonl
experiments/relation_aware_pipeline/out_santos/baseline_sgddg.jsonl
experiments/relation_aware_pipeline/out_santos/autoddg_0.jsonl
experiments/relation_aware_pipeline/out_santos/autoddg_2.jsonl
experiments/relation_aware_pipeline/out_santos/autoddg_4.jsonl
experiments/relation_aware_pipeline/figures/README.md
experiments/info_channel_analysis/README.md

web search: 
web search: https://arxiv.org/html/2502.01050v3
**我的判断：主线还在，但重心已经偏向“怎样安置新增事实”，而不是“怎样选择输入，让搜索更好”。下一步选 C：以固定预算下的输入选择与组合为主攻；A 暂作诊断结论，B 只做有条件的小实验。**

AutoDDG 本身已经包含内容与语义双 profiler、UFD/SFD 分工，以及人工质量验证。因此，双通道和忠实性提升是基础，新的贡献需要落到“选择什么信息、怎样组合、对搜索和用户理解有什么收益”。[AutoDDG 原文](https://arxiv.org/html/2502.01050v3)

**1. 对照原始目的打分**

✓ 表示已实质推进，△ 表示只完成部分，✗ 表示目前缺少完成证据。

| 原始目标 | 判定 | 把关意见 |
|---|---|---|
| profiler 精进 | ✓ | 事实正确性已有实质进展；但“提取更准”尚未转化为“更会挑选搜索需要的信息”。继续扩充统计量应降级。 |
| LLM 前的输入选择与组合 | △ | 事实门控解决可信度，尚未解决效用与冗余。**这是下一阶段最该补的主线。** |
| 生成格式改进 | △ | 七槽位是实现进展，尚无证据表明它优于更紧凑的表示；需要同内容、同预算比较。 |
| token 效率 | ✗ | 输出变长是警报，不是完整的成本结论；应统计全流程输入、输出 token，并比较单位预算的收益。图表示暂不优先。 |
| 搜索 benchmark 修正 | 待核验 | 有检索实验不等于完成 benchmark 修正；需要查询意图、相关性标注和候选集审计。 |
| 可读性人工评价 | ✗ | 明确欠账。无支持断言减少不等于更易读、更便于判断数据是否适用。 |
| 长度影响 | △ | 已发现关联，但长度、内容替换、生成波动仍有混杂，尚不能单独归因。 |
| 单表生成界限 | ✗ | 门控不能替代界限研究；需要区分可计算事实、可合理推断语义、必须借助外部证据的信息。 |

**负结果应该降级的是“全量数值事实注入描述”这条具体策略，不是 profiler、事实信息或语义扩展整体。**

还有两个结论必须收窄：

- 词面过滤器只是“词面重叠筛选”，不是语义效用的 oracle。它支持“这批事实对当前查询的直接词面覆盖极低”，不能证明事实没有检索价值。
- dense 目前表现为“对所测扰动不敏感”。没有检出下降，不等于普遍免疫；它也可能同时忽略了有用与无用的变化。

**2. 下一步选 C：预算约束下的输入选择与组合**

核心问题改成：

> 在相同 token 预算下，列名、代表值、全局统计和语义线索怎样选择与组合，才能保住有用语义并改善搜索？

这里的选择发生在生成前，不能使用测试查询或相关性标签。语义联想继续开放，但“可能的用途、相关概念”与“已经证实的数据属性”应在表达上区分。

| 方案 | 对原始目的的贡献 | 证据与成本判断 | 定位 |
|---|---|---|---|
| A：忠实性与检索的张力 | 能解释现象，但本身不改善搜索 | 成本低；现有证据不足以支持普遍、不可避免的张力 | 阶段性诊断 |
| B：事实独立字段索引 | 若用户确有时间、地域、数值约束需求，则有价值 | 独立字段可以减轻正文干扰，却不会自动创造相关信号；工程成本中等 | 条件性支线 |
| C：输入选择与组合 | 直接覆盖导师主线、长度与效率 | 最大损失出现在再生成环节，支持优先研究输入如何改变语义内容；可复用现有流程 | **主攻** |

C 的证据支持的是“值得优先检验”，还不是“肯定有效”。先做三个小条件即可：

1. 修 bug 后的原流程；
2. 相同预算下，选择列名、代表值和少量有解释价值的统计；
3. 相同预算下的简单截断或随机选择对照。

检验选择是否优于单纯缩短。另设一个**冻结基础 SFD、只追加事实**的诊断条件，隔离再生成造成的变化。不要一开始同时开发复杂选择器、图表示和新检索器。

B 只有在查询审计发现真实的结构化约束需求时才值得做；不能为了证明事实有用，专门制造一组偏爱事实的查询来替换原 benchmark。

**3. 论文主张**

建议把目标主张定为：

> **面向数据集搜索，在固定 token 预算下选择并组合可靠的表内容证据、保留有用的语义扩展，比全量事实注入更能兼顾检索效果与元数据可读性。**

这是待验证主张，当前不能当作已证结论。

**负结果是配角：用于排除“更准确、更多事实自然带来更好检索”的设计假设，并解释选择机制的必要性。**如果后续选择机制仍无收益，就应诚实收缩为实证诊断论文；那也意味着原定“改善搜索”的目标尚未完成，不能靠改写标题宣告完成。

**4. 接下来两周的优先级**

| 时间 | 优先任务 | 应得到的结果 |
|---|---|---|
| 第 1—2 天 | 审计现有 32 个查询及相关性标注；区分主题、用途、实体、覆盖范围、数值约束需求 | 明确当前 benchmark 测得到什么；修正有独立依据的问题，并保留修正前结果 |
| 第 2—5 天 | 做上述最小输入选择实验；控制预算、生成配置，关键条件少量重复生成 | 判断收益来自选择、缩短，还是偶然措辞变化 |
| 第 3—6 天，并行 | 启动可读性人工小评测：约 10—15 表、至少两位评价者、匿名随机展示 | 分别评价易读性、信息充分性、简洁性及判断适用性的帮助；不要合成一个掩盖差异的分数 |
| 第 6—8 天 | 固定信息内容比较两档长度；比较等预算的输入组合 | 分开估计长度影响和信息构成影响，报告 token 成本 |
| 第 8—9 天 | 用同批表做“单表界限”案例标注 | 列出可计算、合理推断、需外部证据三类内容；用具体反例约束主张 |
| 第 10 天 | 综合检索、人工评价、成本与失败案例作继续／停止决策 | 决定是否扩大样本；B 只有通过查询需求检查才进入下一轮 |

有三件事尤其值得补上：

- **查清再生成丢掉了什么。** 最大损失在这里，应检查主题词、同义表达、用途线索、实体信息是否消失或被稀释，而不只数新增事实匹配了多少查询。
- **把 dense 当诊断对照。** 比较各查询排名变化，看它是否保住了 BM25 丢失的语义匹配；当前没有理由直接把 dense 当作改进方案。
- **把单表界限接到实际输入选择上。** 例如来源、采集目的或单位不能唯一确定时，需要什么最小外部元数据？先用案例说明边界，不急着建立“可识别性理论”。

现有 32 查询已经参与了诊断与方案选择，后续结果应标为探索性；扩大结论前，需要未用于调方案的独立查询验证。

**5. 最后把关**

**有偏航苗头：你们正在从“为搜索选择信息”滑向“替已经提取的事实寻找用处”；现在该保留 profiler 的正确性成果，把主攻重新拉回输入选择、语义保留和用户评价。**
tokens used
14,570
**我的判断：主线还在，但重心已经偏向“怎样安置新增事实”，而不是“怎样选择输入，让搜索更好”。下一步选 C：以固定预算下的输入选择与组合为主攻；A 暂作诊断结论，B 只做有条件的小实验。**

AutoDDG 本身已经包含内容与语义双 profiler、UFD/SFD 分工，以及人工质量验证。因此，双通道和忠实性提升是基础，新的贡献需要落到“选择什么信息、怎样组合、对搜索和用户理解有什么收益”。[AutoDDG 原文](https://arxiv.org/html/2502.01050v3)

**1. 对照原始目的打分**

✓ 表示已实质推进，△ 表示只完成部分，✗ 表示目前缺少完成证据。

| 原始目标 | 判定 | 把关意见 |
|---|---|---|
| profiler 精进 | ✓ | 事实正确性已有实质进展；但“提取更准”尚未转化为“更会挑选搜索需要的信息”。继续扩充统计量应降级。 |
| LLM 前的输入选择与组合 | △ | 事实门控解决可信度，尚未解决效用与冗余。**这是下一阶段最该补的主线。** |
| 生成格式改进 | △ | 七槽位是实现进展，尚无证据表明它优于更紧凑的表示；需要同内容、同预算比较。 |
| token 效率 | ✗ | 输出变长是警报，不是完整的成本结论；应统计全流程输入、输出 token，并比较单位预算的收益。图表示暂不优先。 |
| 搜索 benchmark 修正 | 待核验 | 有检索实验不等于完成 benchmark 修正；需要查询意图、相关性标注和候选集审计。 |
| 可读性人工评价 | ✗ | 明确欠账。无支持断言减少不等于更易读、更便于判断数据是否适用。 |
| 长度影响 | △ | 已发现关联，但长度、内容替换、生成波动仍有混杂，尚不能单独归因。 |
| 单表生成界限 | ✗ | 门控不能替代界限研究；需要区分可计算事实、可合理推断语义、必须借助外部证据的信息。 |

**负结果应该降级的是“全量数值事实注入描述”这条具体策略，不是 profiler、事实信息或语义扩展整体。**

还有两个结论必须收窄：

- 词面过滤器只是“词面重叠筛选”，不是语义效用的 oracle。它支持“这批事实对当前查询的直接词面覆盖极低”，不能证明事实没有检索价值。
- dense 目前表现为“对所测扰动不敏感”。没有检出下降，不等于普遍免疫；它也可能同时忽略了有用与无用的变化。

**2. 下一步选 C：预算约束下的输入选择与组合**

核心问题改成：

> 在相同 token 预算下，列名、代表值、全局统计和语义线索怎样选择与组合，才能保住有用语义并改善搜索？

这里的选择发生在生成前，不能使用测试查询或相关性标签。语义联想继续开放，但“可能的用途、相关概念”与“已经证实的数据属性”应在表达上区分。

| 方案 | 对原始目的的贡献 | 证据与成本判断 | 定位 |
|---|---|---|---|
| A：忠实性与检索的张力 | 能解释现象，但本身不改善搜索 | 成本低；现有证据不足以支持普遍、不可避免的张力 | 阶段性诊断 |
| B：事实独立字段索引 | 若用户确有时间、地域、数值约束需求，则有价值 | 独立字段可以减轻正文干扰，却不会自动创造相关信号；工程成本中等 | 条件性支线 |
| C：输入选择与组合 | 直接覆盖导师主线、长度与效率 | 最大损失出现在再生成环节，支持优先研究输入如何改变语义内容；可复用现有流程 | **主攻** |

C 的证据支持的是“值得优先检验”，还不是“肯定有效”。先做三个小条件即可：

1. 修 bug 后的原流程；
2. 相同预算下，选择列名、代表值和少量有解释价值的统计；
3. 相同预算下的简单截断或随机选择对照。

检验选择是否优于单纯缩短。另设一个**冻结基础 SFD、只追加事实**的诊断条件，隔离再生成造成的变化。不要一开始同时开发复杂选择器、图表示和新检索器。

B 只有在查询审计发现真实的结构化约束需求时才值得做；不能为了证明事实有用，专门制造一组偏爱事实的查询来替换原 benchmark。

**3. 论文主张**

建议把目标主张定为：

> **面向数据集搜索，在固定 token 预算下选择并组合可靠的表内容证据、保留有用的语义扩展，比全量事实注入更能兼顾检索效果与元数据可读性。**

这是待验证主张，当前不能当作已证结论。

**负结果是配角：用于排除“更准确、更多事实自然带来更好检索”的设计假设，并解释选择机制的必要性。**如果后续选择机制仍无收益，就应诚实收缩为实证诊断论文；那也意味着原定“改善搜索”的目标尚未完成，不能靠改写标题宣告完成。

**4. 接下来两周的优先级**

| 时间 | 优先任务 | 应得到的结果 |
|---|---|---|
| 第 1—2 天 | 审计现有 32 个查询及相关性标注；区分主题、用途、实体、覆盖范围、数值约束需求 | 明确当前 benchmark 测得到什么；修正有独立依据的问题，并保留修正前结果 |
| 第 2—5 天 | 做上述最小输入选择实验；控制预算、生成配置，关键条件少量重复生成 | 判断收益来自选择、缩短，还是偶然措辞变化 |
| 第 3—6 天，并行 | 启动可读性人工小评测：约 10—15 表、至少两位评价者、匿名随机展示 | 分别评价易读性、信息充分性、简洁性及判断适用性的帮助；不要合成一个掩盖差异的分数 |
| 第 6—8 天 | 固定信息内容比较两档长度；比较等预算的输入组合 | 分开估计长度影响和信息构成影响，报告 token 成本 |
| 第 8—9 天 | 用同批表做“单表界限”案例标注 | 列出可计算、合理推断、需外部证据三类内容；用具体反例约束主张 |
| 第 10 天 | 综合检索、人工评价、成本与失败案例作继续／停止决策 | 决定是否扩大样本；B 只有通过查询需求检查才进入下一轮 |

有三件事尤其值得补上：

- **查清再生成丢掉了什么。** 最大损失在这里，应检查主题词、同义表达、用途线索、实体信息是否消失或被稀释，而不只数新增事实匹配了多少查询。
- **把 dense 当诊断对照。** 比较各查询排名变化，看它是否保住了 BM25 丢失的语义匹配；当前没有理由直接把 dense 当作改进方案。
- **把单表界限接到实际输入选择上。** 例如来源、采集目的或单位不能唯一确定时，需要什么最小外部元数据？先用案例说明边界，不急着建立“可识别性理论”。

现有 32 查询已经参与了诊断与方案选择，后续结果应标为探索性；扩大结论前，需要未用于调方案的独立查询验证。

**5. 最后把关**

**有偏航苗头：你们正在从“为搜索选择信息”滑向“替已经提取的事实寻找用处”；现在该保留 profiler 的正确性成果，把主攻重新拉回输入选择、语义保留和用户评价。**
