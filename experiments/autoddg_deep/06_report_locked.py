"""Assemble reproducible Chinese report; unavailable runs remain explicitly pending."""
import json
from pathlib import Path
B=Path(__file__).resolve().parent
R=B/'results'
complete=(R/'eval/l1l2.json').exists()
eval_path=R/'eval'/('l1l2.json' if complete else 'l1l2_reference.json')
ev=json.loads(eval_path.read_text())
noise=json.loads((R/'paired/baseline__r1__subset30__vs__baseline__r2__subset30.json').read_text())['rows']
audit=json.loads((R/'audit'/('F2_L1_L2__r1__subset30.json' if complete else 'F2__r1__subset30.json')).read_text())
lines=['# L1/L2：程序化锁定 UFD→SFD（subset30）','',
'状态：'+('实验完成。' if complete else '**实现完成，实验未完成。** 首个 L1 API 请求在域名解析阶段失败（httpcore.ConnectError，Errno 8），未取得新响应；L1/L2 指标与 p 值不得用旧结果代填。'),'',
'## 实现与实验边界','',
'- AutoDDG 仓库仅新增 description/search_locked.py、fact_coverage.py；原包既有文件不改。实验仓库的 05_fact_audit.py 导入抽出的同一归一/覆盖逻辑。',
'- L1：Dataset Overview 逐字复制 F2 缓存中的 UFD，LLM 仅输出主题、应用、概念、关键词与上下文补充。事实输入不变。',
'- L2：与 L1 复用同一请求缓存，按 selected_sfd 的 ID 检查，遗漏事实原渲染文本追加到 Verified Coverage and Measures；不调用修补 LLM。删除补充尾部腾出预算后再检查，直至全部覆盖或明确报错。',
'- 一个缓存事实 F027 将 3.9998 渲染为 4，原审计拒绝该小数近似；追加同一 fact.value 的精确端点，不放宽审计。',
'- 上游直接读取 F2/r1 和其所指向的 F1 profile，不调用 profiling/topic/semantic/UFD，也不因新 commit 使旧缓存失效。指纹绑定源记录、profile、配置和实现。',
'- 固定 subset30：29 张有成功的 F2 SFD；fbb92a2a-4b64-4ac8-ba0f-8d608bfc47ec 原 F2 因 212272 输入 token 超过 128000 而失败，新臂继承相同失败和 title+description 回退。',
'- F2 SFD 原本无显式输出上限。新臂每表采用对应 F2 SFD 的 whitespace 词数上限（含标题），复制与修补均计入；LLM 生成足量补充后按词边界截取，L1/L2 最终词数必须等于 F2（否则报错），不填充无证据内容。等长单位是 whitespace 词数，不是模型 token 数；尾部截取可能影响可读性。',
'- 因新臂取消了 LLM 的 overview/事实专段生成，并增加预算指令，L2−F2 是所实现机制的总干预效果，不能严称为仅“重新生成丢信息”的无混杂因果估计。L2−L1 使用完全相同的补充文本，能隔离覆盖修补及其预算挤占效应。',
'- 评测沿用 03_index_eval.py：BM25 k1=1.5、b=0.75、lower().split()，sfd/per_query，NDCG@10/20；配对随机化 10000 次、bootstrap 10000 次、seed=9。all 是主结果，with_table/listed 是敏感性结果。32 个原查询中 CSV 判断池有正相关的为 25 个，with_table 为 18 个，listed 为 8 个。仅对 30 张生成；all 评测保留既有全判断池，其余文档统一原描述回退。','',
'## 噪声底：baseline r2 − r1','',
'|池|指标|n|r1|r2|Δ|p|95% CI|','|---|---|---:|---:|---:|---:|---:|---|']
for r in noise:
 lines.append(f"|{r['setting'].split('|')[-1]}|{r['metric']}|{r['n_queries']}|{r['mean_a']:.6f}|{r['mean_b']:.6f}|{r['delta']:+.6f}|{r['p_randomization']:.4f}|[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}]|")
lines+=['','噪声底只是一对重复的经验差值，不是可靠的显著性阈值；超过它仍需看配对 p 值、区间和方向一致性。','', '## 四组对照','', 'Δ 均为目标臂减参照臂；L1 vs L2 按 L2−L1 报告。']
for a,b in [('bugfix','L2'),('F2','L2'),('F2','L1'),('L1','L2')]:
 lines+=['',f'### {b} − {a}','']
 if not complete:
  lines+=['待运行：均值、逐查询差值、p 值及置信区间尚不可计算。']
  continue
 rs=json.loads((R/f'paired/{a}__r1__subset30__vs__{b}__r1__subset30.json').read_text())['rows']
 lines+=['|池|指标|Δ|p|95% CI|胜/平/负|噪声底绝对值|','|---|---|---:|---:|---|---|---:|']
 for r in rs:
  n=next(n for n in noise if (n['setting'],n['metric'])==(r['setting'],r['metric']))
  lines.append(f"|{r['setting'].split('|')[-1]}|{r['metric']}|{r['delta']:+.6f}|{r['p_randomization']:.4f}|[{r['ci95_low']:+.6f},{r['ci95_high']:+.6f}]|{r['wins']}/{r['ties']}/{r['losses']}|{abs(n['delta']):.6f}|")
 for subset in ('all','with_table','listed'):
  lines+=['',f'逐查询（{subset}）：','', '|query|Δ@10|Δ@20|噪声Δ@10|噪声Δ@20|','|---|---:|---:|---:|---:|']
  rr=[next(r for r in rs if r['setting'].endswith('|'+subset) and r['metric']==m) for m in ('ndcg@10','ndcg@20')]
  nn=[next(r for r in noise if r['setting'].endswith('|'+subset) and r['metric']==m) for m in ('ndcg@10','ndcg@20')]
  for q in rr[0]['per_query_delta']:
   vals=[r['per_query_delta'][q] for r in rr+nn]
   lines.append('|'+q+'|'+'|'.join(f'{v:+.6f}' for v in vals)+'|')
lines+=['','## 事实保留率','', '|臂|成功表|SFD 保留|保留率|','|---|---:|---|---:|']
for a in ('F2','L1','L2'):
 s=audit['summary'].get(a)
 if s:
  f=s['sfd_retention'];lines.append(f"|{a}|{s['n_tables']}|{f['n_kept']}/{f['n_facts']}|{f['rate']:.4%}|")
 else: lines.append(f'|{a}|待运行|待运行|待运行|')
lines+=['','该口径是旧审计的锚点保留：列表只要求 min(2,n) 个值，不等于全部列表值、列值关系或语义正确性得到证明。L2 的精确端点来自程序证据，若旧 unsupported-number 检查只看 LLM prompt，可能将其标成未支持，需结合 locked_repairs 和原 fact.value 核对。','',
'## 结论与续跑','',
'验证：实验测试集 125 项通过，新增的等长输出/不足拒绝两项测试亦通过（含 30 表缓存上的离线预算/覆盖检查；此检查不生成实验臂结果）。AutoDDG 子仓库独立实现提交为 `270a5e0`；主仓库 git add/commit 被 .git/index.lock 的 Operation not permitted 阻止，当前会话不能完成主仓库提交。', '',
'程序结构保证 overview 不经 LLM 重写；成功返回的 L2 保证 selected_sfd 通过现有覆盖谓词。检索是否改善、覆盖检查值多少，必须等真实 L1/L2 生成和配对结果，当前不能下结论。','',
'执行 `zsh experiments/autoddg_deep/run_l1l2.sh`：source ~/.zshrc（不显示 key），L1 在线、L2 缓存模式，随后评测/配对/审计并重建本报告。脚本遇错立即停止；不会扩大生成表集。缓存及 results/ 默认被 gitignore 忽略，报告与脚本纳入版本控制。']
(B/'REPORT_L1L2.md').write_text('\n'.join(lines)+'\n')
print('wrote REPORT_L1L2.md; complete=',complete)
