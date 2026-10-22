"""D1 local analysis: within-pair discrimination, neighbour leakage, unsupported numbers, cost.

discrimination  for every confusion triple (q, better, worse) from
                configs/confusion_pairs.json: rank both tables with BM25 over q's
                judged pool (03_index_eval settings, sfd view, all subset) and
                count better above worse. Equal-grade positive pairs: ranks only.
leakage         content words (lower case, >= 3 letters) in an arm's SFD that do
                not occur in the table's own F2 SFD prompt but do occur in the
                neighbour text D1 was shown. F2 gets the same count against the
                same neighbour text as the coincidence baseline.
numbers         numbers in the SFD that are not in the table's own F2 SFD prompt
                (the neighbour block excluded), as in 05_fact_audit.
cost            prompt / completion tokens of D1's SFD calls (gpt-4o-mini list
                price $0.15 / $0.60 per 1M tokens).

Writes results/audit/D1_discrimination.json.
"""
import importlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE_DIR, HERE, RESULTS_DIR, select_benchmark  # noqa: E402
ev = importlib.import_module('03_index_eval')

ARMS = ['bugfix__r1__subset30', 'F2__r1__subset30', 'D1__r1__subset30']
PRICE_IN, PRICE_OUT = 0.15e-6, 0.60e-6
WORD = re.compile(r'[a-z]{3,}')
NUM = re.compile(r'(?<![\w.])-?\d+(?:\.\d+)?(?![\w])')


def norm_nums(text):
    return set(NUM.findall(re.sub(r'(?<=\d),(?=\d{3}\b)', '', text)))


def main():
    pairs = json.loads((HERE/'configs/confusion_pairs.json').read_text())
    queries, qrels, corpus = select_benchmark(csv_only=True)
    out = {'triples': [], 'equal_grade': [], 'summary': {}}
    docs = {a: ev.arm_documents(a, 'sfd', corpus)[0] for a in ARMS}
    ranks = {}
    for a in ARMS:
        for q in {t['query'] for t in pairs['triples'] + pairs['equal_grade_pairs']}:
            ranked = ev.rank_bm25(queries[q], sorted(qrels[q]), docs[a])
            ranks[a, q] = {d: r + 1 for r, d in enumerate(ranked)}
    for t in pairs['triples']:
        row = dict(t, pool=len(qrels[t['query']]))
        for a in ARMS:
            rk = ranks[a, t['query']]
            row[a] = {'rank_better': rk[t['better']], 'rank_worse': rk[t['worse']],
                      'correct': rk[t['better']] < rk[t['worse']]}
        out['triples'].append(row)
    for t in pairs['equal_grade_pairs']:
        row = dict(t, pool=len(qrels[t['query']]))
        for a in ARMS:
            row[a] = [ranks[a, t['query']][d] for d in t['tables']]
        out['equal_grade'].append(row)
    for a in ARMS:
        n = len(out['triples'])
        out['summary'][a] = {'discrimination_rate': sum(r[a]['correct'] for r in out['triples']) / n if n else None,
                             'n_triples': n,
                             'equal_grade_mean_rank': (sum(sum(r[a]) for r in out['equal_grade']) /
                                                       (2 * len(out['equal_grade'])) if out['equal_grade'] else None)}

    ids = (HERE/'configs/subset30.txt').read_text().split()
    audit, usage = [], {'prompt_tokens': 0, 'completion_tokens': 0, 'calls': 0, 'cache_hits': 0}
    for i in ids:
        d1 = json.loads((CACHE_DIR/f'gen/D1/r1/{i}.json').read_text())
        if d1['status'] != 'ok':
            continue
        f2 = json.loads((CACHE_DIR/f'gen/F2/r1/{i}.json').read_text())
        own = f2['sfd_prompt'].lower()
        own_words, nb_words = set(WORD.findall(own)), set(WORD.findall(d1['discriminative_neighbour_text'].lower()))
        own_nums = norm_nums(f2['sfd_prompt'])
        row = {'dataset_id': i, 'neighbour': d1['discriminative_neighbour']['neighbour'],
               'length_delta': d1['discriminative_length_delta']}
        for arm, rec in (('F2', f2), ('D1', d1)):
            w = set(WORD.findall(rec['sfd'].lower()))
            row[f'{arm}_neighbour_only_words'] = sorted((w - own_words) & nb_words)
            row[f'{arm}_unsupported_numbers'] = sorted(norm_nums(rec['sfd']) - own_nums)
        row['D1_mentions_similar'] = bool(re.search(r'similar dataset|unlike|other dataset', d1['sfd'], re.I))
        audit.append(row)
        for c in d1['calls']['sfd']:
            u = c.get('usage') or {}
            usage['calls'] += 1; usage['cache_hits'] += bool(c['cache_hit'])
            usage['prompt_tokens'] += u.get('prompt_tokens', 0); usage['completion_tokens'] += u.get('completion_tokens', 0)
    usage['usd'] = usage['prompt_tokens'] * PRICE_IN + usage['completion_tokens'] * PRICE_OUT
    out['audit'] = audit
    out['usage'] = usage
    out['audit_summary'] = {
        arm: {'tables_with_leak': sum(bool(r[f'{arm}_neighbour_only_words']) for r in audit),
              'leaked_words': sum(len(r[f'{arm}_neighbour_only_words']) for r in audit),
              'tables_with_unsupported_numbers': sum(bool(r[f'{arm}_unsupported_numbers']) for r in audit),
              'unsupported_numbers': sum(len(r[f'{arm}_unsupported_numbers']) for r in audit)}
        for arm in ('F2', 'D1')}
    out['audit_summary']['D1_mentions_similar'] = sum(r['D1_mentions_similar'] for r in audit)
    path = RESULTS_DIR/'audit/D1_discrimination.json'
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ('summary', 'audit_summary', 'usage')}, indent=1))
    for r in out['triples']:
        print(r['query'], r['better'][:8], '>', r['worse'][:8], ' '.join(
            f"{a.split('__')[0]}:{r[a]['rank_better']}/{r[a]['rank_worse']}" for a in ARMS), 'pool', r['pool'])
    for r in out['equal_grade']:
        print('eq', r['query'], [t[:8] for t in r['tables']], ' '.join(f"{a.split('__')[0]}:{r[a]}" for a in ARMS))
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
