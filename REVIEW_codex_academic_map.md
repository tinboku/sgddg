我会先核对 AutoDDG 原文、附录和检索基准，再沿五条脉络查证关键论文；会明确区分已有贡献、可主张的新颖点，以及仍需实验支持的判断。
web search: 
web search: https://arxiv.org/html/2502.01050v1
web search: 
web search: Doc2Query-- When Less is More 2023 filtering ...
web search: 
web search: 'Appendix B'
web search: 
web search: "data-to-text" "planning" "LLM" content planning ...
web search: 
web search: "dataset retrieval" "EI" ...
已核实一个重要边界：AutoDDG 的 SFD 模板已经要求原样保留初始描述，也已有同义词扩展。因此，“保留事实”和“增加查询词”本身不能作为新颖点；需要落到可验证的保留机制，以及表格证据如何约束扩展。另一个关键区别是：面向问答的表检索与开放数据集发现，不能直接混用任务结论。
web search: 
web search: "dataset retrieval" "explicit" "implicit" EI ...
web search: 
web search: "Haoxiang Zhang" "ECIR" description ...
web search: 
web search: AutoDDG ECIR 2024 2025 automated description generation Freire ...
exec
/bin/zsh -lc "rg -n 'ECIR|\\bEI\\b|content planning|内容规划' --glob '*.md' --glob '*.bib' --glob 'AGENTS.md' ." in /Users/mo/Desktop/sgddg-github
 succeeded in 71ms:
./TASK_codex_academic_map.md:9:1. **内容选择与规划（Content Selection & Planning）**：data-to-text 的内容选择脉络（传统 pipeline 系统、neural 规划、LLM 时代的 planning）。AutoDDG 附录 B 的内容规划在这条线上的位置。"程序化锁定保留"对照这条线的什么？
./TASK_codex_academic_map.md:12:4. **表格检索与元数据检索（Table Retrieval）**：DTR、DataFinder、TURL、EI(Dataset retrieval)、AutoDDG 自己的 ECIR 前作、SPLADE/多字段检索在表格上的应用。benchmark 生态（NTCIR-DDG、ECIR-DDG、WWW 2018）各是什么定位。
./REVIEW_codex_improvements.md:20:web search: 'ECIR' in https://arxiv.org/html/2502.01050v3
./REVIEW_codex_improvements.md:24:1. **生成环：把 UFD→SFD 的事实保留变成程序约束。**保留现有两阶段生成，让 LLM 只生成 SFD 的主题、关键词和应用补充，`overview` 由程序原样复制 UFD；另将本轮已选事实记录为带 ID 的清单，检查其在 SFD 中的覆盖情况，遗漏时定点修补，而非全篇重写。先比较“当前流程／只锁 UFD／锁 UFD＋事实覆盖检查”，保持输出预算相同。预期贡献是阻断你们已定位的信息损失，**并不假定保住所有事实就能提高检索**。原版其实已经在 prompt 中要求保留初始描述，因此改进点是可执行保证；内容规划也有 data-to-text 先例。[AutoDDG 附录 B](https://arxiv.org/html/2502.01050v3)、[Content Selection and Planning](https://ojs.aaai.org/index.php/AAAI/article/view/4668)。**成本：1–2 人日；复制与检查无需额外 LLM 调用，修补按需调用。**
./REVIEW_codex_improvements.md:50:14. **评测环：扩大语料与查询来源，建立真正的外部验证。**下一周优先接 ECIR-DDG，先完成数据可用性检查和端到端小规模跑通；随后扩展至其完整协议。论文报告它有 120 查询、1,015 表，但查询来自有限任务，划分应按原始任务分组。再选 [WWW 2018 关键词表检索集](https://github.com/iai-group/table-retrieval)做跨来源验证；[DataFinder](https://arxiv.org/abs/2305.16636)的 392 条专家评测查询适合科研数据需求，但须先确认对应原始数据是否能经过双 profiler，不能直接视为即插即用替代。[AutoDDG benchmark 说明](https://arxiv.org/html/2502.01050v3)。所有方法固定候选集合，报告 NDCG、Recall、逐查询胜负和配对区间；32 查询已用于诊断，应将最终外部集留作确认。**成本：3–5 人日启动；完整生成成本随表数增长，贡献是提高结论可信度。**
./REVIEW_codex_improvements.md:59:1. **生成环：把 UFD→SFD 的事实保留变成程序约束。**保留现有两阶段生成，让 LLM 只生成 SFD 的主题、关键词和应用补充，`overview` 由程序原样复制 UFD；另将本轮已选事实记录为带 ID 的清单，检查其在 SFD 中的覆盖情况，遗漏时定点修补，而非全篇重写。先比较“当前流程／只锁 UFD／锁 UFD＋事实覆盖检查”，保持输出预算相同。预期贡献是阻断你们已定位的信息损失，**并不假定保住所有事实就能提高检索**。原版其实已经在 prompt 中要求保留初始描述，因此改进点是可执行保证；内容规划也有 data-to-text 先例。[AutoDDG 附录 B](https://arxiv.org/html/2502.01050v3)、[Content Selection and Planning](https://ojs.aaai.org/index.php/AAAI/article/view/4668)。**成本：1–2 人日；复制与检查无需额外 LLM 调用，修补按需调用。**
./REVIEW_codex_improvements.md:85:14. **评测环：扩大语料与查询来源，建立真正的外部验证。**下一周优先接 ECIR-DDG，先完成数据可用性检查和端到端小规模跑通；随后扩展至其完整协议。论文报告它有 120 查询、1,015 表，但查询来自有限任务，划分应按原始任务分组。再选 [WWW 2018 关键词表检索集](https://github.com/iai-group/table-retrieval)做跨来源验证；[DataFinder](https://arxiv.org/abs/2305.16636)的 392 条专家评测查询适合科研数据需求，但须先确认对应原始数据是否能经过双 profiler，不能直接视为即插即用替代。[AutoDDG benchmark 说明](https://arxiv.org/html/2502.01050v3)。所有方法固定候选集合，报告 NDCG、Recall、逐查询胜负和配对区间；32 查询已用于诊断，应将最终外部集留作确认。**成本：3–5 人日启动；完整生成成本随表数增长，贡献是提高结论可信度。**
./PLAN_REVIEW_autoddg_deep.md:21:- 修正版 NTCIR-DDG（标注质量审计：AutoDDG 论文自曝 ECIR TP 率仅 9.87%，Zenodo 404）+ 人工可读性评价 + 长度/token 受控曲线（已知 confound：205 词 vs 1185 词渲染差异主导 ELO 差距）+ 单表信息界限（二代 failure_taxonomy 77/2186 ≈ 3.5% 不可回元）。
./REVIEW_codex_v11.md:81:其中，**23/233 的改写已正确**：它是原 ECIR 被标正例的抽样确认比例，不需要再改；但它不能作为 NTCIR 标签错误率的证据。[论文 §4.2](https://arxiv.org/html/2502.01050v1#S4.SS2)
./REVIEW_codex_v11.md:201:其中，**23/233 的改写已正确**：它是原 ECIR 被标正例的抽样确认比例，不需要再改；但它不能作为 NTCIR 标签错误率的证据。[论文 §4.2](https://arxiv.org/html/2502.01050v1#S4.SS2)
./experiments/agent_metadata_gen/pilot/AGENT_AND_METADATA_SPEC.md:166:2. **保留 UFD/SFD 是为了可比性。** AutoDDG 的 BM25 ECIR nDCG@20：
./PLAN_autoddg_deep_v1.1.md:43:- "9.87%" 引用要写准：是 ECIR 原标注正例的抽样确认率（23/233），不是修正后 benchmark 的错误率。
./REVIEW_codex_autoddg_deep.md:547:   AutoDDG 报告的是对原 ECIR 正例抽样复核时，233 对中只有 23 对被确认相关；随后作者重新构建了 ECIR-DDG 正例。它既不是修订后 ECIR-DDG 的错误率，也不是 NTCIR 的审计结果。按该抽样分母，它更接近“标正样本的确认比例”，不要直接当作通常定义的 TPR。[原论文 §3.2](https://arxiv.org/html/2502.01050v3#S3.SS2)
./REVIEW_codex_autoddg_deep.md:839:   AutoDDG 报告的是对原 ECIR 正例抽样复核时，233 对中只有 23 对被确认相关；随后作者重新构建了 ECIR-DDG 正例。它既不是修订后 ECIR-DDG 的错误率，也不是 NTCIR 的审计结果。按该抽样分母，它更接近“标正样本的确认比例”，不要直接当作通常定义的 TPR。[原论文 §3.2](https://arxiv.org/html/2502.01050v3#S3.SS2)
./experiments/agent_metadata_gen/pilot/AGENT_METADATA_PRIOR_WORK_AND_DESIGN.md:62:| keywords / synonyms / aliases | 直接扩大 BM25 词面命中 | **V** AutoDDG：SFD 0.877 > UFD 0.836 > Original 0.694（BM25 ECIR nDCG@20）。TACO 的缩写展开在下游提升至多 32% |
./experiments/agent_metadata_gen/pilot/LITERATURE_TO_DESIGN_MATRIX.md:27:| **AutoDDG** `2502.01050` | UFD =「clear, concise, and accurate overview…prioritizing human readability」，由样本 + data-driven profile 生成。SFD =「optimized to enhance the findability」，在 UFD 基础上按 dataset topic 扩出「relevant terms, concepts, synonyms, and keyword variations」，结构为 Dataset Overview / Related Topics / Concepts and Synonyms / Applications-Use Cases / Additional Context，并明言「focus less on readability and more on including all relevant terms」。data-driven profile = 逐列的类型、值分布、唯一性、统计摘要（值域、覆盖期）。semantic profiling = 逐列 LLM 判定 temporal? + resolution、spatial? + resolution、entity type & domain、function/usage（如 Aggregation Key / Measurement），外加 2–3 词的 dataset topic。基准：ECIR-DDG（120 查询 / 169 数据集）、NTCIR-DDG（32 查询 / 615 数据集）。BM25 ECIR nDCG@20：Original 0.694 / UFD 0.836 / **SFD 0.877**。**作者自述限制：hallucination 是「a critical challenge」，描述「necessarily incomplete」**；future work 含 hallucination 检测与缓解、human-in-the-loop。 |
./experiments/agent_metadata_gen/pilot/LITERATURE_TO_DESIGN_MATRIX.md:47:| 2 | **UFD / SFD** | AutoDDG `2502.01050` | 输入：表 + data-driven profile + semantic profile；输出：UFD（可读概览）+ SFD（关键词扩张） | **SFD > UFD 在检索上成立**（BM25 ECIR nDCG@20 0.877 vs 0.836 vs Original 0.694）。SFD 的设计意图就是塞满同义词 | 保留 UFD/SFD 两分，保留固定输出 schema 与长度约束 | AutoDDG 自述 hallucination 是 critical challenge；Faithful-or-Findable 进一步测到"检索最强=忠实最低"。**沿用 SFD 而不约束扩张，等于沿用一个已知会牺牲忠实度的设计** | 保留 UFD/SFD **Direct**；**约束 SFD 扩张 = Hypothesis** |
./experiments/agent_metadata_gen/pilot/LITERATURE_TO_DESIGN_MATRIX.md:57:| 12 | **retrieval benchmark** | AutoDDG 的 ECIR-DDG / NTCIR-DDG | 数据集检索：查询 → 相关数据集，nDCG@k | 这两个是该任务的既有基准，且我们第一篇 SGDDG 已在 NTCIR 上报过数（SFD nDCG@20 = 0.2685） | 作为**主** benchmark | KramaBench 原文是 pipeline 基准，**不能作为唯一主 benchmark**；我们从中派生的 file-retrieval setting 是自制的 | ECIR/NTCIR-DDG **Direct**；KramaBench 派生用法 **Our construction**（须标注） |
./experiments/relation_aware_pipeline/LITERATURE.md:237:### ECIR-DDG — Kolyada et al., ECIR 2025 〔不可获取〕
./experiments/relation_aware_pipeline/LITERATURE.md:289:| ECIR-DDG | 数据 | **无（DOI 404）** | — | 已替换 |
./experiments/relation_aware_pipeline/LITERATURE.md:319:4. **核实 ECIR-DDG 是否有镜像**，或联系作者。
./experiments/agent_metadata_gen/pilot/GENERATION_RESEARCH_RESET.md:162:| **NTCIR-DDG / ECIR-DDG 是否作为主 benchmark** | **是，应作为主 benchmark。** 它们是 AutoDDG 用的同任务基准（ECIR 120 查询/169 数据集；NTCIR 32 查询/615 数据集），我们第一篇已在 NTCIR 上报过 SFD nDCG@20 = 0.2685，可直接对齐 |
./experiments/agent_metadata_gen/pilot/GENERATION_RESEARCH_RESET.md:163:| **KramaBench environment / wildfire** | **只适合开发与 data-lake stress test。** 原文是 end-to-end pipeline 基准（104 挑战 / 1,700 文件 / 6 领域），不是 metadata 基准。继续使用时必须写明这是**我们派生的 file-retrieval setting**，**不能作为唯一主 benchmark**。它的价值在于 37 表里有 22 张同 schema 年份切片——这是 ECIR/NTCIR 没有的压力测试 |
./experiments/relation_aware_pipeline/STUDY.md:361:**ECIR-DDG 不可获取。** 原计划用它（Kolyada et al., ECIR 2025），但 DOI

web search: 
web search: "dataset" "retrieval" "explicit" "implicit" "EI" Luo ...
web search: 
web search: "Faithful or Findable" dataset descriptions ...
web search: 
web search: "AutoDDG" "ECIR" -site:sciencestack.ai -site:alphaxiv.org -site:arxiv.org -site:reddit.com ...
web search: 
web search: "dataset retrieval" "EI" model -site:reddit.com ...
web search: 
web search: "dataset retrieval" "EI" Cheng Chen ...
“EI（Dataset retrieval）”及“AutoDDG 自己的 ECIR 前作”若有论文标题、作者或链接，请补充；目前我尚未核实这两个简称对应的确切文献，会继续完成其余脉络，并将无法确认的对应关系明确标注。
web search: 
web search: "Promptagator" site:openreview.net ...
web search: 
web search: 
web search: 
web search: 
web search: AutoDDG ECIR prior automated dataset descriptions
web search: Pneuma Efficient Table Discovery LLM 2025
web search: Promptagator Few-shot Dense Retrieval From 8 Examples ICLR 2023
web search: dataset retrieval EI model
找到一篇必须主动对照的近期工作：2026 年的《Faithful or Findable?》已在 RDF 数据集搜索中报告忠实性与检索效果不一致。因此，你们的负结果不能单独包装成“首次发现更多事实不一定更好”；更有价值的是解释 AutoDDG 中事实在哪一阶段丢失、保留下来后为何仍不能匹配查询，并用干预实验验证。
web search: 
web search: 
web search: 
web search: 
web search: "Table" "Faithfulness" "PARENT"
web search: "AutoDDG" "ECIR"
web search: "SPLADE: Sparse Lexical and Expansion Model"
web search: "dataset retrieval" "EI"
web search: 
web search: 
web search: 
web search: 
web search: 
web search: "PARENT" "Handling Divergent"
web search: "AutoDDG" SIGMOD 2026
web search: "EI" "dataset search"
web search: "Sketch and Refine" ACL 2021
web search: "Data-to-text Generation with Macro Planning"
web search: 
web search: 
web search: 
web search: 
web search: "SPLADE" "2021" site:arxiv.org/abs
web search: https://arxiv.org/pdf/2502.01050
web search: "Table Retrieval using Natural Language Queries" "sampling" Chen 2020
web search: "LogicNLG" logical natural language generation ACL 2020
web search: 
web search: 'Content Planning'
web search: 
web search: 
web search: 
web search: 
web search: "Ad Hoc Table Retrieval" "60" "1.6"
web search: "Table Search Using a Deep Contextualized Language Model"
web search: "Efficient" "Interpretation" "Dataset Retrieval"
web search: "Overview of the NTCIR-15 Data Search Task"
web search: 
web search: 
web search: 
web search: 
web search: "TaBERT" "content snapshot" ACL 2020
web search: "From doc2query to docTTTTTquery"
web search: "Pneuma" site:arxiv.org/abs/2504.09207
web search: "Enhancing Dataset Search" "EI"
web search: 
web search: https://aclanthology.org/P19-1483/
web search: 
web search: https://krisztianbalog.com/files/www2018-table.pdf
web search: 
web search: https://arxiv.org/pdf/2209.11755
**最有希望的主线是方向 1“内容选择与可验证保留”＋方向 2“证据约束的查询式扩展”。**它们分别对应你们已经观察到的两个瓶颈：事实在 UFD→SFD 中丢失，以及保留下来的事实没有转化为查询可匹配的表达。方向 3 用于控制预算，方向 4 确定任务与基线，方向 5 提供质量约束和机制分析。

以下将你提供的实验结果视为既定出发点，未重新运行实验；文献、方法与发表信息均联网核实。涉及“我们”的新颖点声明是**待实验支持的建议措辞**，不是已经成立的贡献。

先纠正三个容易影响论文定位的边界：

- AutoDDG 原版已对各列**全量行做 profiling**；“全表扫描”本身不是新增贡献。应具体说明 profiler v2 新增哪些事实、如何门控。[AutoDDG §2.1.1](https://arxiv.org/html/2502.01050v3#S2.SS1.SSS1)
- 原文附录 B 展示的是生成模板和提示约束，没有独立、可执行的内容规划器；SFD 模板已经要求原样保留初始描述。[AutoDDG 附录 B](https://arxiv.org/html/2502.01050v3#A2)
- **“AutoDDG 自己的 ECIR 前作”尚未核实成立。**其 ECIR 基准来源是 Chen 等的 ECIR 2020 论文，作者团队不同。“EI”也未能唯一对应到一篇文献，下文明确区分可核实的相邻工作，避免替它编造身份。

**1．内容选择与规划：从“决定说什么”走向“确保选中的内容进入索引”**

| 代表论文 | 年份／venue | 方法与核心发现 |
|---|---|---|
| Reiter，[An Architecture for Data-to-Text Systems](https://aclanthology.org/W07-2315/) | 2007，ENLG | 将数据分析、解释、文档规划、语言实现分开；奠定“事实提取不等于内容选择，内容选择不等于语言实现”的 pipeline 框架。 |
| Puduppully 等，[Data-to-Text Generation with Content Selection and Planning](https://ojs.aaai.org/index.php/AAAI/article/view/4668) | 2019，AAAI | 先生成选取与排序数据记录的内容计划，再条件生成文本；在 RotoWire 上改善自动评价和人工评价。显式规划与端到端训练可以兼容。 |
| Moryossef 等，[Step-by-Step: Separating Planning from Realization](https://aclanthology.org/N19-1236/) | 2019，NAACL | 忠实于输入的符号规划＋神经语言实现；在 WebNLG 上改善可靠性与语义充分性。**符号规划并不是 LLM 时代的新思想。** |
| Puduppully、Lapata，[Data-to-text Generation with Macro Planning](https://doi.org/10.1162/tacl_a_00381) | 2021，TACL | 将实体、事件及其关系组织成段落级计划；在 RotoWire、MLB 上改善事实性、连贯性和流畅性。 |
| Wang 等，[Sketch and Refine，SANA](https://aclanthology.org/2021.findings-acl.427/) | 2021，Findings ACL-IJCNLP | 指针网络从表格选择骨架词，编辑式生成器在骨架硬约束下生成；改善事实覆盖和忠实性。**这是“硬保留”必须主动引用的近邻。** |
| Zhao、Sun，[Poised：Enabling controllable table-to-text generation…](https://doi.org/10.1016/j.knosys.2024.112571) | 2024，Knowledge-Based Systems | 用 prefix-tuned BART 产生计划，再据此选择属性、提示 LLM；在 few-shot Wiki 三领域获得接近或达到 100% 的计划完成率，但仍存在幻觉。 |
| Fierro 等，[Learning to Plan and Generate Text with Citations](https://aclanthology.org/2024.acl-long.615/) | 2024，ACL | 以问题序列作为内容蓝图，将计划与证据归因结合；长答案生成中规划改善引用质量。任务不是表格检索，但“带证据的计划”已有明确先例。 |

**已解决／未解决。**这条线已经充分说明：显式计划、符号计划、骨架硬约束、LLM 按计划生成，都能改善生成控制。尚未由这些工作解决的是：**面对未来未知查询，应该保留哪些表格事实，以及这些事实经多阶段改写后是否仍对检索有用。**传统计划通常针对参考叙事、覆盖率或篇章结构学习，并不直接优化数据集发现。

**AutoDDG 附录 B 的位置。**它更接近固定文档结构加自然语言约束：规定 overview、主题、用途、同义词等栏目，要求保留初始描述。它与“先选事实 ID、再确定顺序、最后检查实现”的显式规划不同。不能写“AutoDDG 没有考虑内容组织”，可以写“其保留要求由提示表达，未形成程序级保留契约”。

**我们的位置。**“程序化锁定保留”至少应分成三层：

1. 程序直接复制 UFD：属于确定性文本组装，保证字面保留。
2. 记录选中事实及来源，检查最终 SFD 覆盖：属于计划执行验证。
3. 依据预算与预期检索价值选择事实：才是检索导向的内容选择。

只做第一层，很可能被认为是实现修复。第三层与第二层结合，才更接近研究贡献。且复制 UFD 只能保证**保留**，不能保证 UFD 本身**正确**。

**新颖点声明：**“我们在 AutoDDG 的多阶段描述生成中引入面向检索的事实选择与可验证保留机制，分离事实选择、语言扩展和最终索引组装。”——**最近邻：AutoDDG、SANA、Poised；必须切割：不声称首次内容规划、首次硬约束或首次提高事实覆盖。**

---

**2．查询生成与文档扩展：证据约束已有先例，空间在表格证据的粒度与约束方式**

| 代表论文 | 年份／venue | 方法与核心发现 |
|---|---|---|
| Nogueira 等，[Document Expansion by Query Prediction](https://arxiv.org/abs/1904.08375) | 2019，arXiv 预印本 | 从文档预测可能查询并追加到文档，离线缓解查询与文档的词汇错配；是 doc2query 的直接起点。 |
| Nogueira、Lin，[From doc2query to docTTTTTquery](https://cs.uwaterloo.ca/~jimmylin/publications/Nogueira_Lin_2019_docTTTTTquery-latest.pdf) | 2019，技术报告／µpublication | 使用 T5 替代原 seq2seq 扩展器，显著改善检索效果。不要误标成某个主会论文。 |
| Alberti 等，[Synthetic QA Corpora Generation with Roundtrip Consistency](https://aclanthology.org/P19-1620/) | 2019，ACL | 从上下文选择答案、生成问题，再用问答模型检验能否恢复答案；过滤后改善 QA 训练。**“生成问题必须由来源支持”已有明确先例。** |
| Gospodinov 等，[Doc2Query--: When Less is More](https://eprints.gla.ac.uk/287683/) | 2023，ECIR | 使用相关性模型过滤生成的劣质查询；发现扩展中的幻觉与无关内容损害检索，过滤可同时改善效果、索引体积和查询效率。 |
| Dai 等，[Promptagator](https://arxiv.org/abs/2209.11755) | 2023，ICLR | 少量示例驱动 LLM 生成任务相关查询，并进行 round-trip 检索一致性过滤，用于训练检索器；证明查询生成需要适配具体检索任务。 |
| Fierro 等，[Learning to Plan and Generate Text with Citations](https://aclanthology.org/2024.acl-long.615/) | 2024，ACL | 问题蓝图与来源归因相连，进一步表明“问题形式＋来源证据”不是空白方向；但输出目标是答案而非索引扩展。 |

**已解决／未解决。**Doc2Query 解决“用用户可能使用的语言描述文档”；Doc2Query-- 解决“不是所有生成查询都值得索引”；round-trip 工作解决生成样本的一致性筛选。

因此，**“证据约束有没有人做过？”答案是有，但需区分约束强度：**

| 约束 | 已有工作中的典型形式 | 不能据此保证什么 |
|---|---|---|
| 来源条件化 | 输入文档后生成查询 | 不能保证查询中每个限定条件都受支持 |
| 相关性过滤 | Doc2Query-- 的 query–document 评分 | 相关不等于逻辑蕴含 |
| 可回答性／回环一致性 | Alberti 的答案恢复、Promptagator 的源文档回检 | 不能保证全部实体、时间、空间、粒度条件正确 |
| 表格级证据验证 | 你们可以实现的列、单元格、统计量及计算结果验证 | 保证范围取决于实际实现的检查器，不能统称“完全忠实” |

**我们的位置。**应把扩展单位设计成“查询式表达＋支持证据＋允许的语义变换”。例如：

- 表格支持“年度失业率”，可以生成相应检索问句及受控同义表达。
- 不能因为主题是就业，就扩展成表中不存在的“月度工资预测”。
- “最早年份是 2010、最晚年份是 2020”不能自动变成“完整覆盖 2010—2020 每一年”。
- 一般用途推测不能冒充表格已经包含的变量或分析结果。

证据验证可以严格限制**事实条件**，同时允许词面不相同的合理表达；否则会把解决 vocabulary mismatch 的能力一并过滤掉。

还要明确：这是**离线 document expansion**，不是在线 query expansion；如果不训练检索器，也不同于 Promptagator 的训练数据生成任务。

**新颖点声明：**“我们为 AutoDDG 构建可追溯到表格事实及计算证据的离线查询式扩展，在固定扩展预算下验证事实条件并保留语义改写，以缓解词汇错配。”——**最近邻：Doc2Query--、Alberti、Promptagator；必须切割：不声称首次 grounded query generation，区别应落在表格证据验证，而非普通相关性过滤。**

---

**3．表格输入选择：同预算是必要比较条件，不是独立新颖点**

| 代表论文 | 年份／venue | 方法与核心发现 |
|---|---|---|
| Yin 等，[TaBERT](https://aclanthology.org/2020.acl-main.745.pdf) | 2020，ACL | 使用与自然语言输入相关的 content snapshot 表示表格内容；实验说明样本内容能改善语义解析，输入不必包含全表。 |
| Chen 等，[Table Search Using a Deep Contextualized Language Model](https://www.cse.lehigh.edu/~brian/pubs/2020/SIGIR/) | 2020，SIGIR | 在 BERT 长度限制下选择行、列、单元格并编码，结合表检索特征改善排序。**输入选择直接服务表检索，早于 LLM。** |
| Parikh 等，[ToTTo](https://aclanthology.org/2020.emnlp-main.89/) | 2020，EMNLP | 给定表格和高亮单元格生成一句描述；高亮信息把“选择哪些内容”外显化，但选择通常由任务输入提供。 |
| Sui 等，[Table Meets LLM](https://www.microsoft.com/en-us/research/publication/table-meets-llm-can-large-language-models-understand-structured-table-data-a-benchmark-and-empirical-study/?locale=zh-cn%2Fbibtex%2F) | 2024，WSDM | 系统研究格式、顺序、分隔标记及提示设计；这些输入选择显著影响表格理解和下游任务。 |
| Sui 等，[TAP4LLM](https://aclanthology.org/2024.findings-emnlp.603.pdf) | 2024，Findings EMNLP | 统一表格采样、知识增强、token 分配与序列化；比较规则、语义、混合采样，覆盖 QA、验证及 ToTTo。**预算分配和采样策略比较已有系统研究。** |
| Balaka 等，[Pneuma](https://arxiv.org/abs/2504.09207) | 2025，SIGMOD／PACMMOD | 使用 schema 描述与行级信息构建可检索表示，结合全文与向量搜索；说明检索表示需要兼顾结构与内容。 |

**已解决／未解决。**已有研究充分证明“格式与采样会影响效果”，TAP4LLM 也已经研究 token allocation。你们不能把“比较不同采样方式”或“在相同 token 数下比较”本身作为算法创新。

更有意义的未解决问题是：**离线建立可服务未知查询的表格描述时，怎样在代表性样本、全局统计与稀有但有区分力的事实之间分配预算？**这与给定问题后选择相关子表有本质区别。

**我们的位置。**profiler v2 可以成为候选事实池，再设计选择器；选择目标应体现检索需求，而不仅是统计代表性。可考虑覆盖主题、实体、时空范围、粒度、单位，并对冗余与 token 成本惩罚。

“同预算”至少要控制：

- LLM 输入 token；同样行数不等于同样输入成本。
- 最终索引文本长度；否则收益可能来自扩展量差异。
- 生成调用及修补成本。
- 检索器、分词和候选集合。

还须保留“全量 profiler 事实直接索引”的低成本基线，检查生成环是否真的必要。

**新颖点声明：**“我们研究未知查询条件下、面向数据集检索的预算受限证据选择，在同等输入与索引预算下协调全局统计、代表性样本和区分性事实。”——**最近邻：TAP4LLM、Chen 等 SIGIR 2020；必须切割：不声称首次表格采样或 token packing，核心差别是离线未知查询和检索目标。**

---

**4．表格检索与元数据检索：必须把任务边界与强基线摆正**

| 代表论文 | 年份／venue | 方法与核心发现；与我们的关系 |
|---|---|---|
| Zhang、Balog，[Ad Hoc Table Retrieval using Semantic Similarity](https://krisztianbalog.com/files/www2018-table.pdf) | 2018，WWW | 结合实体、词及多种语义空间的匹配特征进行学习排序；语义匹配改善关键词表检索。论文已经包含多字段文档检索基线。 |
| Chen 等，[Leveraging Schema Labels to Enhance Dataset Search](https://www.cse.lehigh.edu/~brian/pubs/2020/ECIR/) | 2020，ECIR | 从表格内容生成 schema labels，与元数据匹配分数融合；改善数据集检索。**这是 AutoDDG 所用 ECIR 原始基准的来源，不是同作者前作。** |
| Deng 等，[TURL](https://www.vldb.org/pvldb/vol14/p307-deng.pdf) | 2020，PVLDB 14(3)，VLDB 2021 周期 | 结构感知 Transformer＋Masked Entity Recovery，支持实体链接、列类型、关系抽取等六项任务。它是表理解底座，**不是直接的自然语言数据集检索方法**。 |
| Herzig 等，[Open Domain Question Answering over Tables via Dense Retrieval，DTR](https://aclanthology.org/2021.naacl-main.43.pdf) | 2021，NAACL | TAPAS 双编码器、表—文本预训练及困难负例，改善 NQ-Tables 的检索和最终 QA。目标是找可回答问题的表。 |
| Formal 等，[SPLADE](https://arxiv.org/abs/2107.05720) | 2021，SIGIR | 学习稀疏词汇扩展和权重，兼顾语义匹配与倒排索引；原始方法是文本检索模型，不是表格专用模型。 |
| Viswanathan 等，[DataFinder](https://aclanthology.org/2023.acl-long.573/) | 2023，ACL | 根据研究想法推荐科研数据集；构建训练数据与 392 条专家标注评测查询，训练双编码器。主要对应研究需求与数据集文本信息匹配。 |
| Luo 等，[An Empirical Investigation of Implicit and Explicit Knowledge-Enhanced Methods for Ad Hoc Dataset Retrieval](https://aclanthology.org/2023.findings-emnlp.957/) | 2023，Findings EMNLP | 比较隐式预训练知识和显式实体知识增强；结果支持融合不同匹配信号。是可核实的相邻工作，但**不能未经确认把它命名为“EI”**。 |
| Balaka 等，[Pneuma](https://arxiv.org/abs/2504.09207) | 2025，SIGMOD | LLM 表示、全文／向量混合检索组成端到端发现系统，是比单纯 TURL 更直接的系统级近邻。 |
| Zhang 等，[AutoDDG](https://doi.org/10.1145/3786626) | 2026，SIGMOD／PACMMOD | profiling 与语义增强生成 UFD／SFD，再用于检索；已评测 BM25 和 SPLADE。将生成描述用于检索，以及接入 SPLADE，均已被覆盖。[官方书目信息](https://github.com/VIDA-NYU/AutoDDG) |

关于 **EI**：目前无法核实唯一对应论文。一个可能混淆是 **NTCIR-E1**，这是上述 Luo 等论文使用的英文数据集搜索测试集合名称，不是名为 EI 的算法。正式 related work 应保留这个待核实项，不能用相似简称强行补齐。

**Benchmark 生态：**

| 基准 | 原始定位与来源 | 对这篇论文的合适用途 |
|---|---|---|
| **NTCIR-DDG** | AutoDDG 对政府开放数据搜索资源作适配；32 查询、615 表。原 NTCIR Data Search 面向 Data.gov／e-Stat 的统计数据搜索。[原任务报告](https://research.nii.ac.jp/ntcir/workshop/OnlineProceedings15/pdf/ntcir/01-NTCIR15-OV-DATA-KatoM.pdf) | 直接检验 AutoDDG 改进；查询少且已经用于诊断，不能单独支撑广泛泛化结论。 |
| **ECIR-DDG** | 源自 Chen 等 ECIR 2020 的任务与查询，AutoDDG 重建相关数据集；120 查询、1,015 表，原始查询来自六个任务。 | 同任务外部验证；按原任务分组划分，避免同一需求的改写查询跨训练／测试泄漏。 |
| **WWW 2018** | Wikipedia 关键词表检索，60 条查询，使用分级相关性判断；有标题、表头、表体及页面上下文。[原论文](https://krisztianbalog.com/files/www2018-table.pdf) | 检验跨来源迁移和内容／schema 匹配；不能将其小型百科表的结果等同于大型 CSV 数据集发现。 |
| **NQ-Tables／DTR** | 面向开放域表格问答，检索包含答案的表。 | 适合扩展验证，但问题形态和相关性定义不同。 |
| **DataFinder** | 面向科研想法的数据集推荐，数据对象不限定为可访问 CSV。 | 适合检验意图匹配；需先确认原始数据可取得，不能默认接入 profiler 即可运行。 |

DDG 数量来自 [AutoDDG §3.2、表 1](https://arxiv.org/html/2502.01050v3#S3.SS2)。其中 ECIR-DDG 的 **169 是平均每查询候选表数，不是总表数**；也须区分原论文候选池协议与全库检索协议。

**已解决／未解决。**语义表匹配、多字段检索、schema 扩展、生成元数据和混合检索都已有先例。未解决的具体切口是：原始事实与生成扩展具有不同可信度、长度和匹配作用，怎样组合才能避免长扩展淹没关键证据？

**我们的位置。**主实验应先固定检索器，证明改进来自描述构造；多字段方法作为强基线或补充模块。建议比较原始元数据、schema／原始值、确定性事实文本、AutoDDG SFD、查询扩展及其字段融合。否则，新增字段带来的收益容易被误归因给生成方法。

**新颖点声明：**“我们在固定检索器与候选协议下改进 AutoDDG 的索引表示，并检验可验证事实与生成扩展的独立贡献及互补性。”——**最近邻：AutoDDG、ECIR 2020 schema-label 方法、Pneuma；必须切割：不声称首次语义表检索、多字段检索或 SPLADE 表格应用。**

---

**5．忠实性：你们补充的是“正确事实如何成为检索信号”，不是首次发现质量目标不一致**

| 代表论文 | 年份／venue | 方法与核心发现 |
|---|---|---|
| Gardent 等，[The WebNLG Challenge](https://aclanthology.org/W17-3518/) | 2017，INLG | 将 RDF triples 转换成文本，评估语言质量及语义充分性。严格说是 graph-to-text，不是整张表的摘要或数据集检索。 |
| Dhingra 等，[PARENT：Handling Divergent Reference Texts…](https://aclanthology.org/P19-1483/) | 2019，ACL | 将生成文本、参考文本与源表共同对齐；当参考文本偏离表格时，优于仅依赖参考的 BLEU／ROUGE 与人工评价相关性。 |
| Parikh 等，[ToTTo](https://aclanthology.org/2020.emnlp-main.89/) | 2020，EMNLP | 高亮单元格控制生成内容，人工修订参考句以提高忠实性；流畅的模型输出仍常出现表格不支持的短语。 |
| Chen 等，[LogicNLG](https://aclanthology.org/2020.acl-main.708/) | 2020，ACL | 要求生成可由表格逻辑推出的陈述，研究计数、比较等推理忠实性；说明词面覆盖不等于逻辑正确。 |
| Liu 等，[Towards Faithfulness…from an Entity-centric View](https://ojs.aaai.org/index.php/AAAI/article/download/17583/17390) | 2021，AAAI | 从实体覆盖及幻觉角度建模，使用增强计划改善开放域 table-to-text 忠实性；把遗漏与无依据生成分开处理。 |
| Wang 等，[SANA](https://aclanthology.org/2021.findings-acl.427/) | 2021，Findings ACL-IJCNLP | 骨架硬约束提高源表覆盖与忠实性，但没有证明覆盖提升会提高下游检索。 |
| Terrenzi、Ayvaz，[Faithful or Findable? Evaluating LLM-Generated Metadata for RDF Dataset Search](https://arxiv.org/abs/2607.05970) | 2026，arXiv 预印本 | 联合评价六种元数据生成方式的检索与忠实性；无约束改写检索最强但忠实性最低，profile grounding 改善忠实性，profile-grounded rewriting 提供较均衡折中。**这是你们当前叙事最需要正面对照的近期工作。** |

**已解决／未解决。**已有工作已经证明流畅性、参考相似度、事实覆盖、忠实性是不同指标；近期 RDF 检索工作进一步显示，检索改善可能伴随无依据扩展。因此不能把“更忠实不一定更好搜”写成首创结论。

**你们的负结果具体补充了什么？**

- 对 **SANA、实体中心方法**：补上从生成指标到检索效用的外部有效性检验。它们没有声称覆盖越多检索越好，因此你们不是反驳它们，而是在研究其效果能否传递到新任务。
- 对 **Doc2Query--**：它主要处理低质量扩展造成的损害；你们可以研究**经过验证的真事实也可能没有检索价值**，以及长度、冗余和词汇表达如何介入。
- 对 **AutoDDG**：研究其收益的边界条件与中间损失，而非笼统否定 profiling 或 SFD。
- 对 **Faithful or Findable?**：其主要发现是生成策略间的忠实性—效果权衡；你们应进一步提供**阶段级事实追踪＋干预证据**，解释为什么增加源证据仍未改善最终排名。

尤其要避免把下面三件事合成一个指标：

| 维度 | 应测量什么 |
|---|---|
| 忠实性 | 输出陈述是否得到表格或计算证据支持 |
| 保留／完整性 | 已选事实是否到达 UFD、SFD 和实际索引输入 |
| 检索效用 | 这些事实是否改善相关表相对不相关表的排名 |

“更多事实注入”不自动等于“最终输出更忠实”；“99.9% 词面错配”也不自动等于“99.9% 语义无关”。论文必须给出该百分比的分母、分词方式、匹配单位及适用样本，并分别检验词汇、语义与最终排名变化。

**新颖点声明：**“我们在 AutoDDG 中分离证据正确性、跨阶段保留与检索效用，借助受控干预解释为何增加真实表格事实未必改善数据集检索。”——**最近邻：Faithful or Findable?、Doc2Query--；必须切割：不声称首次发现忠实性与检索效果脱钩，新增贡献必须是可复现的机制解释。**

---

**综合判断：以 1＋2 为方法主线，5 为解释框架，3 和 4 为实验约束**

最能支撑论文的核心问题是：

> **在固定预算下，如何让表格中可验证、具有区分力的事实穿过 AutoDDG 的生成链，并转化为符合用户检索语言的索引表达？**

选择 **1＋2** 的理由：

1. **直接对应已观察瓶颈。**方向 1 修复传递损失，方向 2 处理表达错配；继续单独增加事实提取，未必触及问题。
2. **两个模块能形成必要配合。**只保留事实可能留下大量无用数值；只生成查询可能增加无依据扩展。选择、保留和受证据约束的改写共同工作，才有明确机制。
3. **区别于最近邻更具体。**相对 AutoDDG，新增可执行的事实传递；相对 SANA，目标转向检索表示；相对 Doc2Query--，验证落到表格事实条件；相对 Faithful or Findable?，提供干预而不只报告相关性。
4. **可以用小而有力的实验矩阵证伪。**不必将论文扩张成新的完整搜索引擎。

最关键的主实验是一个固定预算的二因素比较：

|  | 不做查询式扩展 | 做证据约束查询式扩展 |
|---|---|---|
| 原生成流程 | AutoDDG／当前事实注入基线 | 仅扩展 |
| 选择后可验证保留 | 仅保留 | 完整方法 |

再加入无约束 doc2query、相关性过滤式 Doc2Query--、事实直接索引、多字段融合这些竞争解释。联合报告检索结果、事实保留率、无依据陈述率和成本，并在独立查询来源上确认。

**当前最稳妥的贡献定位是“检索导向的证据选择、保留与表达”，而不是“更丰富的 profiler”。**只有当完整方法在预算受控、强基线齐全的条件下改善检索，且干预能解释收益来源，才能把这条主线写成方法论文；若仍无收益，现有结果更适合发展为严格的机制分析与负结果研究。
tokens used
212,127
**最有希望的主线是方向 1“内容选择与可验证保留”＋方向 2“证据约束的查询式扩展”。**它们分别对应你们已经观察到的两个瓶颈：事实在 UFD→SFD 中丢失，以及保留下来的事实没有转化为查询可匹配的表达。方向 3 用于控制预算，方向 4 确定任务与基线，方向 5 提供质量约束和机制分析。

以下将你提供的实验结果视为既定出发点，未重新运行实验；文献、方法与发表信息均联网核实。涉及“我们”的新颖点声明是**待实验支持的建议措辞**，不是已经成立的贡献。

先纠正三个容易影响论文定位的边界：

- AutoDDG 原版已对各列**全量行做 profiling**；“全表扫描”本身不是新增贡献。应具体说明 profiler v2 新增哪些事实、如何门控。[AutoDDG §2.1.1](https://arxiv.org/html/2502.01050v3#S2.SS1.SSS1)
- 原文附录 B 展示的是生成模板和提示约束，没有独立、可执行的内容规划器；SFD 模板已经要求原样保留初始描述。[AutoDDG 附录 B](https://arxiv.org/html/2502.01050v3#A2)
- **“AutoDDG 自己的 ECIR 前作”尚未核实成立。**其 ECIR 基准来源是 Chen 等的 ECIR 2020 论文，作者团队不同。“EI”也未能唯一对应到一篇文献，下文明确区分可核实的相邻工作，避免替它编造身份。

**1．内容选择与规划：从“决定说什么”走向“确保选中的内容进入索引”**

| 代表论文 | 年份／venue | 方法与核心发现 |
|---|---|---|
| Reiter，[An Architecture for Data-to-Text Systems](https://aclanthology.org/W07-2315/) | 2007，ENLG | 将数据分析、解释、文档规划、语言实现分开；奠定“事实提取不等于内容选择，内容选择不等于语言实现”的 pipeline 框架。 |
| Puduppully 等，[Data-to-Text Generation with Content Selection and Planning](https://ojs.aaai.org/index.php/AAAI/article/view/4668) | 2019，AAAI | 先生成选取与排序数据记录的内容计划，再条件生成文本；在 RotoWire 上改善自动评价和人工评价。显式规划与端到端训练可以兼容。 |
| Moryossef 等，[Step-by-Step: Separating Planning from Realization](https://aclanthology.org/N19-1236/) | 2019，NAACL | 忠实于输入的符号规划＋神经语言实现；在 WebNLG 上改善可靠性与语义充分性。**符号规划并不是 LLM 时代的新思想。** |
| Puduppully、Lapata，[Data-to-text Generation with Macro Planning](https://doi.org/10.1162/tacl_a_00381) | 2021，TACL | 将实体、事件及其关系组织成段落级计划；在 RotoWire、MLB 上改善事实性、连贯性和流畅性。 |
| Wang 等，[Sketch and Refine，SANA](https://aclanthology.org/2021.findings-acl.427/) | 2021，Findings ACL-IJCNLP | 指针网络从表格选择骨架词，编辑式生成器在骨架硬约束下生成；改善事实覆盖和忠实性。**这是“硬保留”必须主动引用的近邻。** |
| Zhao、Sun，[Poised：Enabling controllable table-to-text generation…](https://doi.org/10.1016/j.knosys.2024.112571) | 2024，Knowledge-Based Systems | 用 prefix-tuned BART 产生计划，再据此选择属性、提示 LLM；在 few-shot Wiki 三领域获得接近或达到 100% 的计划完成率，但仍存在幻觉。 |
| Fierro 等，[Learning to Plan and Generate Text with Citations](https://aclanthology.org/2024.acl-long.615/) | 2024，ACL | 以问题序列作为内容蓝图，将计划与证据归因结合；长答案生成中规划改善引用质量。任务不是表格检索，但“带证据的计划”已有明确先例。 |

**已解决／未解决。**这条线已经充分说明：显式计划、符号计划、骨架硬约束、LLM 按计划生成，都能改善生成控制。尚未由这些工作解决的是：**面对未来未知查询，应该保留哪些表格事实，以及这些事实经多阶段改写后是否仍对检索有用。**传统计划通常针对参考叙事、覆盖率或篇章结构学习，并不直接优化数据集发现。

**AutoDDG 附录 B 的位置。**它更接近固定文档结构加自然语言约束：规定 overview、主题、用途、同义词等栏目，要求保留初始描述。它与“先选事实 ID、再确定顺序、最后检查实现”的显式规划不同。不能写“AutoDDG 没有考虑内容组织”，可以写“其保留要求由提示表达，未形成程序级保留契约”。

**我们的位置。**“程序化锁定保留”至少应分成三层：

1. 程序直接复制 UFD：属于确定性文本组装，保证字面保留。
2. 记录选中事实及来源，检查最终 SFD 覆盖：属于计划执行验证。
3. 依据预算与预期检索价值选择事实：才是检索导向的内容选择。

只做第一层，很可能被认为是实现修复。第三层与第二层结合，才更接近研究贡献。且复制 UFD 只能保证**保留**，不能保证 UFD 本身**正确**。

**新颖点声明：**“我们在 AutoDDG 的多阶段描述生成中引入面向检索的事实选择与可验证保留机制，分离事实选择、语言扩展和最终索引组装。”——**最近邻：AutoDDG、SANA、Poised；必须切割：不声称首次内容规划、首次硬约束或首次提高事实覆盖。**

---

**2．查询生成与文档扩展：证据约束已有先例，空间在表格证据的粒度与约束方式**

| 代表论文 | 年份／venue | 方法与核心发现 |
|---|---|---|
| Nogueira 等，[Document Expansion by Query Prediction](https://arxiv.org/abs/1904.08375) | 2019，arXiv 预印本 | 从文档预测可能查询并追加到文档，离线缓解查询与文档的词汇错配；是 doc2query 的直接起点。 |
| Nogueira、Lin，[From doc2query to docTTTTTquery](https://cs.uwaterloo.ca/~jimmylin/publications/Nogueira_Lin_2019_docTTTTTquery-latest.pdf) | 2019，技术报告／µpublication | 使用 T5 替代原 seq2seq 扩展器，显著改善检索效果。不要误标成某个主会论文。 |
| Alberti 等，[Synthetic QA Corpora Generation with Roundtrip Consistency](https://aclanthology.org/P19-1620/) | 2019，ACL | 从上下文选择答案、生成问题，再用问答模型检验能否恢复答案；过滤后改善 QA 训练。**“生成问题必须由来源支持”已有明确先例。** |
| Gospodinov 等，[Doc2Query--: When Less is More](https://eprints.gla.ac.uk/287683/) | 2023，ECIR | 使用相关性模型过滤生成的劣质查询；发现扩展中的幻觉与无关内容损害检索，过滤可同时改善效果、索引体积和查询效率。 |
| Dai 等，[Promptagator](https://arxiv.org/abs/2209.11755) | 2023，ICLR | 少量示例驱动 LLM 生成任务相关查询，并进行 round-trip 检索一致性过滤，用于训练检索器；证明查询生成需要适配具体检索任务。 |
| Fierro 等，[Learning to Plan and Generate Text with Citations](https://aclanthology.org/2024.acl-long.615/) | 2024，ACL | 问题蓝图与来源归因相连，进一步表明“问题形式＋来源证据”不是空白方向；但输出目标是答案而非索引扩展。 |

**已解决／未解决。**Doc2Query 解决“用用户可能使用的语言描述文档”；Doc2Query-- 解决“不是所有生成查询都值得索引”；round-trip 工作解决生成样本的一致性筛选。

因此，**“证据约束有没有人做过？”答案是有，但需区分约束强度：**

| 约束 | 已有工作中的典型形式 | 不能据此保证什么 |
|---|---|---|
| 来源条件化 | 输入文档后生成查询 | 不能保证查询中每个限定条件都受支持 |
| 相关性过滤 | Doc2Query-- 的 query–document 评分 | 相关不等于逻辑蕴含 |
| 可回答性／回环一致性 | Alberti 的答案恢复、Promptagator 的源文档回检 | 不能保证全部实体、时间、空间、粒度条件正确 |
| 表格级证据验证 | 你们可以实现的列、单元格、统计量及计算结果验证 | 保证范围取决于实际实现的检查器，不能统称“完全忠实” |

**我们的位置。**应把扩展单位设计成“查询式表达＋支持证据＋允许的语义变换”。例如：

- 表格支持“年度失业率”，可以生成相应检索问句及受控同义表达。
- 不能因为主题是就业，就扩展成表中不存在的“月度工资预测”。
- “最早年份是 2010、最晚年份是 2020”不能自动变成“完整覆盖 2010—2020 每一年”。
- 一般用途推测不能冒充表格已经包含的变量或分析结果。

证据验证可以严格限制**事实条件**，同时允许词面不相同的合理表达；否则会把解决 vocabulary mismatch 的能力一并过滤掉。

还要明确：这是**离线 document expansion**，不是在线 query expansion；如果不训练检索器，也不同于 Promptagator 的训练数据生成任务。

**新颖点声明：**“我们为 AutoDDG 构建可追溯到表格事实及计算证据的离线查询式扩展，在固定扩展预算下验证事实条件并保留语义改写，以缓解词汇错配。”——**最近邻：Doc2Query--、Alberti、Promptagator；必须切割：不声称首次 grounded query generation，区别应落在表格证据验证，而非普通相关性过滤。**

---

**3．表格输入选择：同预算是必要比较条件，不是独立新颖点**

| 代表论文 | 年份／venue | 方法与核心发现 |
|---|---|---|
| Yin 等，[TaBERT](https://aclanthology.org/2020.acl-main.745.pdf) | 2020，ACL | 使用与自然语言输入相关的 content snapshot 表示表格内容；实验说明样本内容能改善语义解析，输入不必包含全表。 |
| Chen 等，[Table Search Using a Deep Contextualized Language Model](https://www.cse.lehigh.edu/~brian/pubs/2020/SIGIR/) | 2020，SIGIR | 在 BERT 长度限制下选择行、列、单元格并编码，结合表检索特征改善排序。**输入选择直接服务表检索，早于 LLM。** |
| Parikh 等，[ToTTo](https://aclanthology.org/2020.emnlp-main.89/) | 2020，EMNLP | 给定表格和高亮单元格生成一句描述；高亮信息把“选择哪些内容”外显化，但选择通常由任务输入提供。 |
| Sui 等，[Table Meets LLM](https://www.microsoft.com/en-us/research/publication/table-meets-llm-can-large-language-models-understand-structured-table-data-a-benchmark-and-empirical-study/?locale=zh-cn%2Fbibtex%2F) | 2024，WSDM | 系统研究格式、顺序、分隔标记及提示设计；这些输入选择显著影响表格理解和下游任务。 |
| Sui 等，[TAP4LLM](https://aclanthology.org/2024.findings-emnlp.603.pdf) | 2024，Findings EMNLP | 统一表格采样、知识增强、token 分配与序列化；比较规则、语义、混合采样，覆盖 QA、验证及 ToTTo。**预算分配和采样策略比较已有系统研究。** |
| Balaka 等，[Pneuma](https://arxiv.org/abs/2504.09207) | 2025，SIGMOD／PACMMOD | 使用 schema 描述与行级信息构建可检索表示，结合全文与向量搜索；说明检索表示需要兼顾结构与内容。 |

**已解决／未解决。**已有研究充分证明“格式与采样会影响效果”，TAP4LLM 也已经研究 token allocation。你们不能把“比较不同采样方式”或“在相同 token 数下比较”本身作为算法创新。

更有意义的未解决问题是：**离线建立可服务未知查询的表格描述时，怎样在代表性样本、全局统计与稀有但有区分力的事实之间分配预算？**这与给定问题后选择相关子表有本质区别。

**我们的位置。**profiler v2 可以成为候选事实池，再设计选择器；选择目标应体现检索需求，而不仅是统计代表性。可考虑覆盖主题、实体、时空范围、粒度、单位，并对冗余与 token 成本惩罚。

“同预算”至少要控制：

- LLM 输入 token；同样行数不等于同样输入成本。
- 最终索引文本长度；否则收益可能来自扩展量差异。
- 生成调用及修补成本。
- 检索器、分词和候选集合。

还须保留“全量 profiler 事实直接索引”的低成本基线，检查生成环是否真的必要。

**新颖点声明：**“我们研究未知查询条件下、面向数据集检索的预算受限证据选择，在同等输入与索引预算下协调全局统计、代表性样本和区分性事实。”——**最近邻：TAP4LLM、Chen 等 SIGIR 2020；必须切割：不声称首次表格采样或 token packing，核心差别是离线未知查询和检索目标。**

---

**4．表格检索与元数据检索：必须把任务边界与强基线摆正**

| 代表论文 | 年份／venue | 方法与核心发现；与我们的关系 |
|---|---|---|
| Zhang、Balog，[Ad Hoc Table Retrieval using Semantic Similarity](https://krisztianbalog.com/files/www2018-table.pdf) | 2018，WWW | 结合实体、词及多种语义空间的匹配特征进行学习排序；语义匹配改善关键词表检索。论文已经包含多字段文档检索基线。 |
| Chen 等，[Leveraging Schema Labels to Enhance Dataset Search](https://www.cse.lehigh.edu/~brian/pubs/2020/ECIR/) | 2020，ECIR | 从表格内容生成 schema labels，与元数据匹配分数融合；改善数据集检索。**这是 AutoDDG 所用 ECIR 原始基准的来源，不是同作者前作。** |
| Deng 等，[TURL](https://www.vldb.org/pvldb/vol14/p307-deng.pdf) | 2020，PVLDB 14(3)，VLDB 2021 周期 | 结构感知 Transformer＋Masked Entity Recovery，支持实体链接、列类型、关系抽取等六项任务。它是表理解底座，**不是直接的自然语言数据集检索方法**。 |
| Herzig 等，[Open Domain Question Answering over Tables via Dense Retrieval，DTR](https://aclanthology.org/2021.naacl-main.43.pdf) | 2021，NAACL | TAPAS 双编码器、表—文本预训练及困难负例，改善 NQ-Tables 的检索和最终 QA。目标是找可回答问题的表。 |
| Formal 等，[SPLADE](https://arxiv.org/abs/2107.05720) | 2021，SIGIR | 学习稀疏词汇扩展和权重，兼顾语义匹配与倒排索引；原始方法是文本检索模型，不是表格专用模型。 |
| Viswanathan 等，[DataFinder](https://aclanthology.org/2023.acl-long.573/) | 2023，ACL | 根据研究想法推荐科研数据集；构建训练数据与 392 条专家标注评测查询，训练双编码器。主要对应研究需求与数据集文本信息匹配。 |
| Luo 等，[An Empirical Investigation of Implicit and Explicit Knowledge-Enhanced Methods for Ad Hoc Dataset Retrieval](https://aclanthology.org/2023.findings-emnlp.957/) | 2023，Findings EMNLP | 比较隐式预训练知识和显式实体知识增强；结果支持融合不同匹配信号。是可核实的相邻工作，但**不能未经确认把它命名为“EI”**。 |
| Balaka 等，[Pneuma](https://arxiv.org/abs/2504.09207) | 2025，SIGMOD | LLM 表示、全文／向量混合检索组成端到端发现系统，是比单纯 TURL 更直接的系统级近邻。 |
| Zhang 等，[AutoDDG](https://doi.org/10.1145/3786626) | 2026，SIGMOD／PACMMOD | profiling 与语义增强生成 UFD／SFD，再用于检索；已评测 BM25 和 SPLADE。将生成描述用于检索，以及接入 SPLADE，均已被覆盖。[官方书目信息](https://github.com/VIDA-NYU/AutoDDG) |

关于 **EI**：目前无法核实唯一对应论文。一个可能混淆是 **NTCIR-E1**，这是上述 Luo 等论文使用的英文数据集搜索测试集合名称，不是名为 EI 的算法。正式 related work 应保留这个待核实项，不能用相似简称强行补齐。

**Benchmark 生态：**

| 基准 | 原始定位与来源 | 对这篇论文的合适用途 |
|---|---|---|
| **NTCIR-DDG** | AutoDDG 对政府开放数据搜索资源作适配；32 查询、615 表。原 NTCIR Data Search 面向 Data.gov／e-Stat 的统计数据搜索。[原任务报告](https://research.nii.ac.jp/ntcir/workshop/OnlineProceedings15/pdf/ntcir/01-NTCIR15-OV-DATA-KatoM.pdf) | 直接检验 AutoDDG 改进；查询少且已经用于诊断，不能单独支撑广泛泛化结论。 |
| **ECIR-DDG** | 源自 Chen 等 ECIR 2020 的任务与查询，AutoDDG 重建相关数据集；120 查询、1,015 表，原始查询来自六个任务。 | 同任务外部验证；按原任务分组划分，避免同一需求的改写查询跨训练／测试泄漏。 |
| **WWW 2018** | Wikipedia 关键词表检索，60 条查询，使用分级相关性判断；有标题、表头、表体及页面上下文。[原论文](https://krisztianbalog.com/files/www2018-table.pdf) | 检验跨来源迁移和内容／schema 匹配；不能将其小型百科表的结果等同于大型 CSV 数据集发现。 |
| **NQ-Tables／DTR** | 面向开放域表格问答，检索包含答案的表。 | 适合扩展验证，但问题形态和相关性定义不同。 |
| **DataFinder** | 面向科研想法的数据集推荐，数据对象不限定为可访问 CSV。 | 适合检验意图匹配；需先确认原始数据可取得，不能默认接入 profiler 即可运行。 |

DDG 数量来自 [AutoDDG §3.2、表 1](https://arxiv.org/html/2502.01050v3#S3.SS2)。其中 ECIR-DDG 的 **169 是平均每查询候选表数，不是总表数**；也须区分原论文候选池协议与全库检索协议。

**已解决／未解决。**语义表匹配、多字段检索、schema 扩展、生成元数据和混合检索都已有先例。未解决的具体切口是：原始事实与生成扩展具有不同可信度、长度和匹配作用，怎样组合才能避免长扩展淹没关键证据？

**我们的位置。**主实验应先固定检索器，证明改进来自描述构造；多字段方法作为强基线或补充模块。建议比较原始元数据、schema／原始值、确定性事实文本、AutoDDG SFD、查询扩展及其字段融合。否则，新增字段带来的收益容易被误归因给生成方法。

**新颖点声明：**“我们在固定检索器与候选协议下改进 AutoDDG 的索引表示，并检验可验证事实与生成扩展的独立贡献及互补性。”——**最近邻：AutoDDG、ECIR 2020 schema-label 方法、Pneuma；必须切割：不声称首次语义表检索、多字段检索或 SPLADE 表格应用。**

---

**5．忠实性：你们补充的是“正确事实如何成为检索信号”，不是首次发现质量目标不一致**

| 代表论文 | 年份／venue | 方法与核心发现 |
|---|---|---|
| Gardent 等，[The WebNLG Challenge](https://aclanthology.org/W17-3518/) | 2017，INLG | 将 RDF triples 转换成文本，评估语言质量及语义充分性。严格说是 graph-to-text，不是整张表的摘要或数据集检索。 |
| Dhingra 等，[PARENT：Handling Divergent Reference Texts…](https://aclanthology.org/P19-1483/) | 2019，ACL | 将生成文本、参考文本与源表共同对齐；当参考文本偏离表格时，优于仅依赖参考的 BLEU／ROUGE 与人工评价相关性。 |
| Parikh 等，[ToTTo](https://aclanthology.org/2020.emnlp-main.89/) | 2020，EMNLP | 高亮单元格控制生成内容，人工修订参考句以提高忠实性；流畅的模型输出仍常出现表格不支持的短语。 |
| Chen 等，[LogicNLG](https://aclanthology.org/2020.acl-main.708/) | 2020，ACL | 要求生成可由表格逻辑推出的陈述，研究计数、比较等推理忠实性；说明词面覆盖不等于逻辑正确。 |
| Liu 等，[Towards Faithfulness…from an Entity-centric View](https://ojs.aaai.org/index.php/AAAI/article/download/17583/17390) | 2021，AAAI | 从实体覆盖及幻觉角度建模，使用增强计划改善开放域 table-to-text 忠实性；把遗漏与无依据生成分开处理。 |
| Wang 等，[SANA](https://aclanthology.org/2021.findings-acl.427/) | 2021，Findings ACL-IJCNLP | 骨架硬约束提高源表覆盖与忠实性，但没有证明覆盖提升会提高下游检索。 |
| Terrenzi、Ayvaz，[Faithful or Findable? Evaluating LLM-Generated Metadata for RDF Dataset Search](https://arxiv.org/abs/2607.05970) | 2026，arXiv 预印本 | 联合评价六种元数据生成方式的检索与忠实性；无约束改写检索最强但忠实性最低，profile grounding 改善忠实性，profile-grounded rewriting 提供较均衡折中。**这是你们当前叙事最需要正面对照的近期工作。** |

**已解决／未解决。**已有工作已经证明流畅性、参考相似度、事实覆盖、忠实性是不同指标；近期 RDF 检索工作进一步显示，检索改善可能伴随无依据扩展。因此不能把“更忠实不一定更好搜”写成首创结论。

**你们的负结果具体补充了什么？**

- 对 **SANA、实体中心方法**：补上从生成指标到检索效用的外部有效性检验。它们没有声称覆盖越多检索越好，因此你们不是反驳它们，而是在研究其效果能否传递到新任务。
- 对 **Doc2Query--**：它主要处理低质量扩展造成的损害；你们可以研究**经过验证的真事实也可能没有检索价值**，以及长度、冗余和词汇表达如何介入。
- 对 **AutoDDG**：研究其收益的边界条件与中间损失，而非笼统否定 profiling 或 SFD。
- 对 **Faithful or Findable?**：其主要发现是生成策略间的忠实性—效果权衡；你们应进一步提供**阶段级事实追踪＋干预证据**，解释为什么增加源证据仍未改善最终排名。

尤其要避免把下面三件事合成一个指标：

| 维度 | 应测量什么 |
|---|---|
| 忠实性 | 输出陈述是否得到表格或计算证据支持 |
| 保留／完整性 | 已选事实是否到达 UFD、SFD 和实际索引输入 |
| 检索效用 | 这些事实是否改善相关表相对不相关表的排名 |

“更多事实注入”不自动等于“最终输出更忠实”；“99.9% 词面错配”也不自动等于“99.9% 语义无关”。论文必须给出该百分比的分母、分词方式、匹配单位及适用样本，并分别检验词汇、语义与最终排名变化。

**新颖点声明：**“我们在 AutoDDG 中分离证据正确性、跨阶段保留与检索效用，借助受控干预解释为何增加真实表格事实未必改善数据集检索。”——**最近邻：Faithful or Findable?、Doc2Query--；必须切割：不声称首次发现忠实性与检索效果脱钩，新增贡献必须是可复现的机制解释。**

---

**综合判断：以 1＋2 为方法主线，5 为解释框架，3 和 4 为实验约束**

最能支撑论文的核心问题是：

> **在固定预算下，如何让表格中可验证、具有区分力的事实穿过 AutoDDG 的生成链，并转化为符合用户检索语言的索引表达？**

选择 **1＋2** 的理由：

1. **直接对应已观察瓶颈。**方向 1 修复传递损失，方向 2 处理表达错配；继续单独增加事实提取，未必触及问题。
2. **两个模块能形成必要配合。**只保留事实可能留下大量无用数值；只生成查询可能增加无依据扩展。选择、保留和受证据约束的改写共同工作，才有明确机制。
3. **区别于最近邻更具体。**相对 AutoDDG，新增可执行的事实传递；相对 SANA，目标转向检索表示；相对 Doc2Query--，验证落到表格事实条件；相对 Faithful or Findable?，提供干预而不只报告相关性。
4. **可以用小而有力的实验矩阵证伪。**不必将论文扩张成新的完整搜索引擎。

最关键的主实验是一个固定预算的二因素比较：

|  | 不做查询式扩展 | 做证据约束查询式扩展 |
|---|---|---|
| 原生成流程 | AutoDDG／当前事实注入基线 | 仅扩展 |
| 选择后可验证保留 | 仅保留 | 完整方法 |

再加入无约束 doc2query、相关性过滤式 Doc2Query--、事实直接索引、多字段融合这些竞争解释。联合报告检索结果、事实保留率、无依据陈述率和成本，并在独立查询来源上确认。

**当前最稳妥的贡献定位是“检索导向的证据选择、保留与表达”，而不是“更丰富的 profiler”。**只有当完整方法在预算受控、强基线齐全的条件下改善检索，且干预能解释收益来源，才能把这条主线写成方法论文；若仍无收益，现有结果更适合发展为严格的机制分析与负结果研究。
