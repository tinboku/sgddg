# Track A Pilot 审计记录（commit 63bd3be，零 API）

日期：2026-10-22 晚；审计人：Hermes（协调者）

## 核实项

| 项目 | 结果 |
|---|---|
| 四臂完整性 | B0/C/Cprime/A 各 17 表，全部 status=ok |
| 模型/温度/采样控制 | 四臂一致：gpt-4o-mini、T=0、sample_size=100、random_state=9 |
| Topic 一致性 | 四臂同表 prompt 前 120 字符完全一致（topic 同源自 manifest、T=0）|
| 数据可复现性 | 17 张表 sha256 全部与记录匹配 |
| 配置可复现性 | arm_{B0,C,Cprime,A}.yaml 全部入库 commit 63bd3be |
| 实际 API 用量 | 68 记录 × 2（UFD+SFD）= 136 次调用；输入 ~332k tok、输出 ~60k tok；估算 **$0.086** |
| 授权记录 | Mo 批准（"批准！"）+ TASK_trackA_pilot.md 硬顶 $1；实际 $0.086 合规 |

## 重要发现（审计新知）

**B0 的"纠错读取"在这 17 张表上没有产生任何实际差异**：B0 与 C 的 read_info 中 n_rows/n_cols/encoding/sep 完全相同，唯一差异是 B0 记录了 `reader: index_col=False` 元数据字段。即：
- 被 CSV 解析 bug 影响的表不在这 17 张 held-out 表里（bug 表在另外的审计子集中）
- 因此 pilot 中 B0 ≈ 原版读取，"纠错读取 vs 原读取"的对比**未实际生效**
-这不影响"caveat 送达失败"的结论（四臂 prompt 输入一致、结论是生成层不写），但意味着 pilot 没有真正测试读取修复的效果

## 定性（按 Mo 指令）

- 结果保留，作为"caveat 缺失是系统性问题（B0 也只有 1/17）+ 选择层改动无法送达生成层"的探索性证据
- **不得**当作 Dataset-Adaptive Profiler 有效性结论（不同实验问题）
- 判定者为 Codex、单次重复、n=17：探索性证据等级
