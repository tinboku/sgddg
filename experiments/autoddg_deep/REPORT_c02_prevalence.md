# C02 Joint Coverage Profiling：问题存在性验证（2026-10-22，零 API）

> C02 判据前置："如果真实数据中很少出现联合覆盖问题，则降低优先级"。本文只验证问题是否存在，不实现 C02。
> 核对由 Claude 完成，不是人工评价。

## 1. 做法

- 表：`configs/subset30.txt` 的 30 张表，bugfix 臂 r1（原 AutoDDG + bugfix，无事实槽）的 UFD/SFD；fbb92a2a 原本就因上下文超长失败，排除，剩 29 张。
- 全表读入（不抽样），找：
  - **时间列**：数值列 >95% 是 1800–2100 的整数；或列名含 date/year/time/period 等、>80% 可解析为日期的列，取年份；
  - **分组列**：非时间的文本列，2–60 个不同值，非空率 >50%（地区、类别、类型等）。
- 对每个（时间列 × 分组列）做 groupby：只看"主要组"（≥5 行且 ≥2% 行），如果某个主要组覆盖的年份数不到全表年份数的 80%，记为**联合覆盖缺口**。
- 再逐句读这些表的 UFD/SFD，判断描述有没有把"全表时间范围"套到缺口组上（例如"各组 X 都覆盖 A–B 年"）。
- 一次性内联分析，没有新增脚本（按研究者要求收尾阶段不写新代码）；逐表结果在 `results/c02/joint_coverage.json`。

## 2. 结果

| 层级 | 表数（共 29） |
|---|---|
| 同时有时间列和分组列 | 20 |
| 至少一对存在联合覆盖缺口（自动） | 8 |
| 其中缺口是真实结构，而不是离群值/脚注/样本太小造成的假象 | 4（2a3cc123、e83c4ac6、10811fc1、4b5bd984） |
| **描述把全表时间范围套到缺口组上（联合覆盖误述）** | **1（2a3cc123）**，另 1 表擦边（4b5bd984） |

排除的 4 表：682835a9（缺口来自 1924 这一个离群年）、dde84e96（全表只有 3 年、70 行）、f359e457（缺口组是脚注"Underreported…"，州 × 年完全覆盖，描述"各州 2015–2026"正确）、be2a86f4（缺口在 rank 列——实际取值是战争名——和关系列"Wife"上，KOREA 组不会有 1910 年出生，属于语义上必然的缺口，不算误导）。e83c4ac6 的 rank 对也同理排除，但它的墓地 × 出生年缺口是真实的。

## 3. 案例原文

**案例 1（确认误述）2a3cc123 未婚女性生育率**

- 实际：全表 1970–2015（38 个年份）；`Hispanic`、`Non-Hispanic white` 只有 1990–2015（26 年），`Asian or Pacific Islander total` 只有 2000–2015（16 年）。
- UFD 原文："This dataset provides insights into the birth rates of unmarried women across various age groups and **racial or Hispanic origins from 1975 to 2015**. It includes 38 unique years and categorizes data into six distinct age groups and six racial or ethnic categories."
- 问题：把"种族/西语裔来源"和"1975–2015"绑在一起说，读者会以为 6 个种族类别都能做 1975 年起的趋势分析；实际上其中 3 类在 1990 或 2000 年之前没有数据。SFD 的 Dataset Overview 原样复制这句。

**案例 2（擦边）4b5bd984 New Orleans 311 请求**

- 实际：全表 Date Created 2019–2026；`Traffic Signals/Signs/Striping/Streetlights` 只有 2019–2023，`Roads/Drainage`、`Parks & Parkways` 只到 2024。
- UFD 原文："…including information on request types such as **traffic signals**, trash collection, and property maintenance. … The dataset spans from **January 2, 2019, to March 6, 2023**…"
- 问题：这里的范围本身就错了（来自 100 行样本，全表到 2026-10-20），属于 C04/F2 已处理的"样本范围 vs 全表事实"问题；描述没有声称各类型都覆盖到 2026，所以不算联合覆盖误述。

**案例 3（有真实缺口但描述没犯错）e83c4ac6 Arkansas 退伍军人墓地**

- 实际：两处墓地 FAYETTEVILLE NATIONAL CEMETERY 的出生年 1835–2018，ARKANSAS STATE VETERANS CEMETERY 只有 1941–2018（年份覆盖 23%）。
- UFD 原文："…including personal details such as first names, middle names, last names, birth and death dates, and military service information. It features a rich semantic profile, with temporal data covering birth and death dates…"
- 描述完全没给时间范围，也没有按墓地分开说，所以缺口存在但没有被误述。

## 4. 结论

- 联合覆盖缺口在真实表中**确实存在**（29 表中 4 表有实质缺口），但 AutoDDG 描述**很少**对它做出可检验的错误陈述：**确认误述 1/29 表**，擦边 1 表。原因是描述通常只给一个全表时间范围，很少把时间和具体分组绑在一起说。
- 这 30 张表上更常见的时间类错误是"用 100 行样本的范围代替全表范围"（10811fc1、4b5bd984、be2a86f4 等），这已由 F2 的全表事实槽覆盖，与联合覆盖无关。
- **按 C02 判据：问题在本数据上少见 → 降低优先级。** 如要继续，需要换成以"地区 × 年份"面板数据为主的表集重新测发生率，这一步需要研究者决定。

## 5. 局限

- 时间列、分组列靠启发式识别，会漏（例如 1ff67fc2 的 ESTYMD 列名不含关键字，未被识别）；分组列上限 60 个不同值，城市、邮编等高基数列未检查。
- 80% 覆盖阈值、主要组阈值是事前设定的单一取值，没有做敏感性分析。
- 误述判断由 Claude 逐条阅读完成，只看 UFD 和 SFD 的 Dataset Overview 段。
