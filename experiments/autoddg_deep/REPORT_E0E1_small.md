# E0/E1 小规模对照报告（30 张表）

日期：2026-10-21。只跑了下面列出的 30 张表，没有跑全量。

## 一句话结论

在这 30 张表上，**看不到 bug 修复带来的检索信号**。主设定（SFD / BM25 / per_query 池 / 全语料）下 bugfix 比 baseline r1 高 ΔNDCG@10 = +0.0017（胜/平/负 3/22/0，p = 0.25），但同一臂两次生成之间的差（baseline r2 − r1）是 +0.0051，比这个还大；bugfix 对 baseline r2 是 −0.0034。少数设定里 bugfix 看起来有提升，但 baseline r2 相对 r1 也出现了几乎同样的提升，说明那是生成噪声，不是修复的效果。

## 1. 子集与选表规则

- 脚本：`01_select_subset.py`，名单 `configs/subset30.txt`，元数据（规则、候选数、读不出的表）`configs/subset30.json`。
- 规则：从评测语料（32 条 test 查询的判定池，限有 CSV 的数据集，共 483 个）里取下载成功（manifest status 为 ok/truncated，198 个）并且 `load_table` 能读的表（191 个，排除了 7 张读不出的），按 id 排序后用 `pandas.Series.sample(n=30, random_state=9)` 抽样。抽样时不排除宽表。
- 这 30 张表在 49 个"查询–表"对里有判定，其中 15 对是相关（L1/L2）；有 21 张表对任何查询都不相关。

| id | 标题（截断） | 列数 | 下载状态 | 生成 | Coverage 行差异数 | 被判定/相关的查询数 |
|---|---|---|---|---|---|---|
| `0cbb09dc` | NCHS - VSRR Quarterly provisional estimates for infant | 10 | ok | ok | 1 | 2/2 |
| `10811fc1` | Closed-End Fund Information | 10 | ok | ok | 1 | 1/0 |
| `14f56ea0` | Campaign Finance - Status Of Voluntary Expenditure Ceil | 4 | ok | ok | 1 | 1/0 |
| `19610d2e` | Affordable Housing Pipeline | 68 | ok | ok | 38 | 1/0 |
| `1bf0f2e9` | Eviction Notices | 47 | ok | ok | 17 | 1/1 |
| `1cf07054` | Monthly Page Views to CDC.gov | 5 | ok | ok | 2 | 1/0 |
| `1ff67fc2` | Financial Institution Office Locations | 30 | ok | ok | 16 | 1/0 |
| `201460a5` | Traffic Collision Data from 2010 to Present | 18 | truncated | ok | 7 | 1/0 |
| `20e29a5b` | Calls for Service 2014 | 21 | truncated | ok | 4 | 1/0 |
| `2a208933` | SFO Airport Monthly Utility Consumption for Natural Gas | 7 | ok | ok | 2 | 1/0 |
| `2a3cc123` | NCHS - Birth Rates for Unmarried Women by Age, Race, an | 4 | ok | ok | 1 | 1/0 |
| `2f1aabc5` | Calls for Service 2019 | 21 | truncated | ok | 4 | 1/1 |
| `3e23114a` | NCHS - Drug Poisoning Mortality by State: United States | 18 | ok | ok | 12 | 6/3 |
| `4b5bd984` | 311 OPCD Calls (2012-Present) | 19 | truncated | ok | 5 | 1/0 |
| `4e4c5dbb` | EWDD - Worksource Center Program Performance Outcomes | 7 | ok | ok | 5 | 3/1 |
| `51ea24c0` | Port Of Los Angeles - Adopted Budget 2010 - 2014 | 4 | ok | ok | 0 | 1/0 |
| `53acdac4` | U.S. Life Expectancy at Birth by State and Census Tract | 6 | ok | ok | 3 | 1/0 |
| `5f04aaa3` | Calls for Service 2017 | 21 | truncated | ok | 4 | 1/1 |
| `62acddae` | NCHS - Teen Birth Rates for Age Group 15-19 in the Unit | 9 | ok | ok | 6 | 2/0 |
| `682835a9` | SFEC Form 3410B - Quarterly Report for Permit Consultan | 18 | ok | ok | 1 | 1/0 |
| `7b64babd` | Percentage of Adults Who Report Driving After Drinking | 4 | ok | ok | 2 | 1/0 |
| `b1efbf4b` | Consolidated State Performance Report, 2009-10 | 228 | ok | ok | 114 | 2/0 |
| `be2a86f4` | Gravesite locations of Veterans and beneficiaries in RH | 25 | ok | ok | 2 | 3/0 |
| `c4ca6131` | Right of Way Exception Codes | 2 | ok | ok | 0 | 2/0 |
| `c4e2ce61` | Restaurant Scores - LIVES Standard | 22 | ok | ok | 11 | 1/1 |
| `dcefd0c1` | 2009 VHA Facility Quality and Safety Report - Patient S | 20 | ok | ok | 8 | 1/1 |
| `dde84e96` | Expenditure Lobbyists Registration and Monthly Disclosu | 12 | ok | ok | 1 | 2/0 |
| `e83c4ac6` | Gravesite locations of Veterans and beneficiaries in AR | 25 | ok | ok | 1 | 1/0 |
| `f359e457` | VSRR Provisional Drug Overdose Death Counts | 12 | ok | ok | 4 | 6/3 |
| `fbb92a2a` | Job Patterns For Minorities And Women In Private Indust | 599 | ok | **error（三臂）** | — | 1/0 |

"生成"一列三臂相同。"Coverage 行差异数"是 baseline 与 bugfix 的 basic_profile 有几行不同。

## 2. 三臂完成情况

三臂的参数全部锁定：gpt-4o-mini，UFD 100 词，description/topic temperature 0，SFD temperature 实际都是 0（baseline 不传，用 client 默认值 0.0；bugfix 显式传 0.0），抽样 100 行 random_state=9，sequential 语义模式。

| 臂 | 成功 | error | 退化输入 |
|---|---|---|---|
| baseline r1 | 29 | 1 | 1 |
| baseline r2 | 29 | 1 | 1 |
| bugfix r1 | 29 | 1 | 1 |

- 三臂失败的是同一张表 `fbb92a2a`（599 列），在 UFD 调用时 `context_length_exceeded`（212,272 tokens，上限 128k）。按预注册规则处理：三臂都记为 error，评测时用"标题 + 原描述"这一相同的退化输入。另一张宽表 `b1efbf4b`（228 列）三臂都成功了。
- 子集里没有读不出的表，因为它们在选表时已经被排除了。
- 评测语料里子集以外的 453 个数据集，三臂都用"标题 + 原描述"，所以三臂之间只有这 29 张表的文本不同。
- **输入确实不同**：29 张表里有 27 张的 basic_profile 在两臂之间不同，一共 273 行，全部是 Coverage 那一句（例如 `0cbb09dc` 的 Rate 列，原版写 `from 0 to 112.35`，修复后是 `from 1.82 to 112.35`）。没有 Coverage 以外的差异。
- **输出噪声很大**：在 temperature 0 下，baseline r1 和 r2 的 UFD 只有 1/29 一字不差，SFD 是 0/29。bugfix 和 baseline r1 相比也是 UFD 1/29、SFD 0/29 一致。也就是说，同一臂重跑一次，文本的变化程度已经和换成 bugfix 差不多。

## 3. 评测结果（BM25，NDCG@10）

命令：`03_index_eval.py --arms baseline__r1__subset30,baseline__r2__subset30,bugfix__r1__subset30,human --dataset-ids configs/subset30.txt --tag e0e1_subset30`，以及 `04_paired_test.py`（双侧配对随机化检验 10,000 次，bootstrap 10,000 次，seed 9）。32 条查询里，有 25 条在 CSV 语料里还留有相关数据集，所以 n=25。

**主设定**是 SFD / BM25 / per_query / all，选它是因为这是 AutoDDG 的 NTCIR 协议、README 的主命令，而且 SFD 是面向检索的那个视图。需要说明：这次任务没有事先指定主设定，这是我按协议选的；其余设定都列在后面，没有删掉任何一个。

### 3.1 主设定逐查询结果

25 条查询里只有 3 条的 NDCG@10 在任意两臂之间有变化，其余 22 条完全一样。原因是这 22 条查询的判定池里没有子集表，或者子集表的文本变了但排名没变。

| 查询 | 判定数 | 其中子集表 | 子集相关表 | baseline r1 | bugfix − r1 | r2 − r1 | bugfix − r2 |
|---|---|---|---|---|---|---|---|
| DS2-E-0032 | 32 | 3 | 1 | 0.1799 | +0.0249 | +0.1100 | −0.0851 |
| DS2-E-0050 | 19 | 2 | 0 | 0.3313 | +0.0149 | +0.0149 | 0.0000 |
| DS2-E-0051 | 29 | 4 | 2 | 0.2823 | +0.0027 | +0.0027 | 0.0000 |

bugfix 相对 r1 的 3 个"胜"里，有 2 个 r2 相对 r1 也一模一样地出现了；剩下 1 个（0032）r2 的提升比 bugfix 还大。

### 3.2 配对检验汇总（BM25，NDCG@10，all = 全语料）

| 视图/池 | bugfix − r1 | 胜/平/负 | p | r2 − r1（噪声底） | bugfix − r2 | 胜/平/负 |
|---|---|---|---|---|---|---|
| SFD / per_query（主） | +0.0017 | 3/22/0 | 0.25 | +0.0051 | −0.0034 | 0/24/1 |
| SFD / global | +0.0044 | 4/21/0 | 0.12 | +0.0048 | −0.0004 | 1/22/2 |
| UFD / per_query | +0.0032 | 1/23/1 | 1.00 | 0.0000 | +0.0032 | 1/23/1 |
| UFD / global | −0.0004 | 0/24/1 | 1.00 | +0.0068 | −0.0072 | 1/22/2 |

NDCG@20 的情况相同：主设定 bugfix − r1 = +0.0014（3/21/1，p = 0.25），噪声底 0.0027。所有设定里只有一个 p < 0.05，是 SFD / global / all 的 NDCG@20（+0.0042，8/15/2，p = 0.017）；但同一设定下 bugfix − r2 = +0.0026，p = 0.19。考虑到一共比较了 24 个设定 × 2 个指标，出现一个 p < 0.05 在预期之内，不能当作信号。

dense 和 hybrid（per_query / all，NDCG@10）：SFD dense −0.0008，SFD hybrid +0.0100（噪声底 0.0088），UFD dense +0.0046，UFD hybrid +0.0112。所有 p 都 ≥ 0.37。

### 3.3 只看子集表的设定（listed，n=8）

只在 30 张子集表里排序时，数字看起来更大（例如 UFD / BM25 / global / listed：bugfix − r1 = +0.240，4/3/1，p = 0.12）。但同一设定下 r2 − r1 也有 +0.067，bugfix − r2 = +0.173（p = 0.12）。只有 8 条查询，其中好几条只有 1–2 张被判定的表，NDCG 很容易在 0 和 1 之间跳。我把它当作探索性结果，不作为结论依据。

## 4. 噪声底及其影响

- 文本层面：temperature 0 下两次生成几乎从来不会完全一样（UFD 1/29，SFD 0/29），所以 baseline r1 和 r2 是两份不同的描述。
- 检索层面：主设定下 |r2 − r1| = 0.0051，大于 |bugfix − r1| = 0.0017。在 4 个 BM25 / all 设定里，有 3 个的噪声底 ≥ bugfix 效应。
- `04_paired_test.py` 的 `effective` 判定要有一个前提才可靠：r1/r2 只有一对，噪声底就只是一个点估计，可能非常接近 0（比如 UFD / per_query 的 NDCG@10 噪声底是 0.0000）。这种情况下 `effective=True` 几乎没有信息量。表里有些 `effective=True` 就是这样来的，我没有拿它们当证据。比较可靠的看法是同时看 bugfix − r1 和 bugfix − r2：如果修复真的有效，这两个差值都应该为正。实际结果里这两个差值的符号不一致。
- 统计功效的问题：30 张表只占语料的约 6%，而且 25 条查询里有 22 条的分数在所有臂之间完全不变。所以这个子集对检索指标几乎没有分辨力。即使修复有一个很小的真实效应，这个规模也检测不到。

## 5. 结论

1. 修复确实改变了输入：29 张可用的表里有 27 张、共 273 行 Coverage 句被纠正（比如 `0 to X` 改成真实的下界）。
2. 但在第一个对照里，**检索指标上没有可见的修复信号**。主设定的差值（+0.0017）比两次 baseline 运行之间的差（0.0051）还小，bugfix 对 r2 甚至略负，所有主要设定 p ≥ 0.12。
3. 这是"没有检测到"，不是"确认没有效应"。限制它的主要是 LLM 生成噪声（temperature 0 也不确定），以及子集太小、能影响到的查询太少。

## 6. 产物与改动

- 新增 `01_select_subset.py`（选表）和 `05_analyse_small.py`（文本差异 + 逐查询差值）。
- `02_generate.py` 新增 `--dataset-ids FILE`：只跑名单里的表，输出写到 `<arm>__<run_id>__<名单文件名>.jsonl`，不会覆盖全量文件。生成缓存 run-id 仍然是 r1/r2，以后跑全量时这 30 张会直接命中缓存。
- `03_index_eval.py` 新增 `--dataset-ids FILE`，会自动加一个 `listed` 子集。原有参数和默认行为都没变。
- 结果文件：`results/descriptions/*__subset30.jsonl`、`results/eval/e0e1_subset30{.json,_summary.csv,_analysis.json,_analysis_global.json}`、`results/paired/*subset30*`；日志在 `logs/*subset30*.log`、`logs/paired_*.log`、`logs/eval_e0e1_subset30.log`。
- AutoDDG 原包文件没有改动（相对 3c59e1c 只有新增文件）。

### 运行期间的情况

- 有另一个会话同时在 `external/AutoDDG` 里开发 M2/F1/F2（提交了 profiling_v2，并在 20:14–20:15 改了 `autoddg_deep.py` 和 `prompts_v2.yaml`）。我核对过：bugfix r1 在 19:5x–20:07 之间跑完，用的是 `ec8922d`，早于这些改动，而且这段时间新增的提交都只加了文件，没有改 bugfix 用到的代码。baseline 只用原包文件，这些文件没有被改过，所以 baseline 记录里的 commit 号（`a99af0c`）不同不影响结果。
- 上次会话留下的 profile 进程（PID 34570）在本次运行中自己结束了，它没有调用 LLM。
- 这些 F1/F2 相关文件不是本任务的产物，我没有提交它们。
