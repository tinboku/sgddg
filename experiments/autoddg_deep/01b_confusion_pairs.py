"""Most-confusable neighbour per subset30 table (local, no API) + confusion triples.

Neighbour of table t: BM25 (03_index_eval settings) with t's bugfix document
(title + SFD; title + description on failure) as the query over the other 29
bugfix documents. Candidates are tiered by the qrels, highest tier first, and
the top BM25 candidate of the highest non-empty tier is taken:
  tier 2  t and the candidate are both positive (rel >= 1) for some query
  tier 1  both judged for some query, at least one positive
  tier 0  no shared judged query (pure BM25)
The pure BM25 top-1 is stored too, so the label influence on the choice is visible.

Confusion triples (q, better, worse) are taken from the unordered neighbour pairs
with a shared judged query where the two relevance grades differ; equal-grade
positive pairs have no correct order and are listed separately. At most 10 pairs
(tier 2 first, then BM25 score).

Writes configs/confusion_pairs.json (all 30 neighbours, pairs, triples) and
configs/confusion_pairs.txt (the pair list, human readable).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, RESULTS_DIR, read_jsonl, select_benchmark  # noqa: E402
import importlib
ev = importlib.import_module('03_index_eval')

MAX_PAIRS = 10


def documents(ids, corpus):
    rows = {r['dataset_id']: r for r in read_jsonl(RESULTS_DIR/'descriptions/bugfix__r1__subset30.jsonl')}
    docs = {}
    for i in ids:
        title = corpus[i].get('title') or ''
        r = rows[i]
        docs[i] = f"{title} {r['sfd']}" if r.get('status') == 'ok' and r.get('sfd') else f"{title} {corpus[i].get('description') or ''}"
    return docs


def main():
    ids = (HERE/'configs/subset30.txt').read_text().split()
    assert len(ids) == len(set(ids)) == 30
    queries, qrels, corpus = select_benchmark(csv_only=True)
    docs = documents(ids, corpus)
    from rank_bm25 import BM25Okapi
    shared = {}  # (a, b) -> list of (qid, rel_a, rel_b)
    for q in sorted(queries):
        j = qrels[q]
        for a in ids:
            for b in ids:
                if a != b and a in j and b in j and max(j[a], j[b]) >= 1:
                    shared.setdefault((a, b), []).append((q, j[a], j[b]))

    def tier(a, b):
        s = shared.get((a, b), [])
        return 2 if any(ra >= 1 and rb >= 1 for _, ra, rb in s) else (1 if s else 0)

    neighbours = {}
    for t in ids:
        others = [o for o in ids if o != t]
        bm = BM25Okapi([ev.tokenize(docs[o]) for o in others], k1=ev.BM25_K1, b=ev.BM25_B)
        scores = dict(zip(others, map(float, bm.get_scores(ev.tokenize(docs[t])))))
        order = sorted(others, key=lambda o: (-scores[o], o))
        best = max(tier(t, o) for o in others)
        pick = next(o for o in order if tier(t, o) == best)
        neighbours[t] = {'neighbour': pick, 'tier': best, 'bm25': scores[pick],
                         'bm25_rank': order.index(pick) + 1, 'bm25_top1': order[0],
                         'shared_queries': shared.get((t, pick), [])}

    pairs = {}
    for t, n in neighbours.items():
        key = tuple(sorted((t, n['neighbour'])))
        if n['tier'] and key not in pairs:
            pairs[key] = {'tables': list(key), 'tier': n['tier'], 'bm25': n['bm25'],
                          'shared_queries': shared[key]}
    chosen = sorted(pairs.values(), key=lambda p: (-p['tier'], -p['bm25']))[:MAX_PAIRS]
    triples, ties = [], []
    for p in chosen:
        a, b = p['tables']
        for q, ra, rb in p['shared_queries']:
            if ra == rb:
                ties.append({'query': q, 'tables': [a, b], 'rel': ra})
            else:
                better, worse = (a, b) if ra > rb else (b, a)
                triples.append({'query': q, 'better': better, 'worse': worse, 'rel_better': max(ra, rb), 'rel_worse': min(ra, rb)})
    out = {'rule': __doc__.strip().splitlines()[0], 'neighbours': neighbours, 'pairs': chosen,
           'triples': triples, 'equal_grade_pairs': ties}
    (HERE/'configs/confusion_pairs.json').write_text(json.dumps(out, indent=1))
    lines = ['# pair_a pair_b tier bm25 shared_queries(q:rel_a/rel_b)']
    for p in chosen:
        lines.append(f"{p['tables'][0]} {p['tables'][1]} {p['tier']} {p['bm25']:.2f} " +
                     ','.join(f'{q}:{ra}/{rb}' for q, ra, rb in p['shared_queries']))
    (HERE/'configs/confusion_pairs.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'neighbours: tier counts', {k: sum(n['tier'] == k for n in neighbours.values()) for k in (0, 1, 2)},
          'differs from BM25 top1:', sum(n['neighbour'] != n['bm25_top1'] for n in neighbours.values()))
    print(f'triples={len(triples)} equal_grade={len(ties)}')


if __name__ == '__main__':
    main()
