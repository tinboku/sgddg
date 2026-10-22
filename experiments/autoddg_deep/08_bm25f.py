"""BM25F with separate fields for the subset30 tables, weights tuned by 5-fold CV (local, no API).

Usage:
    python -I 08_bm25f.py

Fields per dataset (subset30 tables; every other judged dataset has only title and
narrative = original description, identical in every arm):
    title      NTCIR title
    narrative  bugfix SFD (fallback: original description)
    facts      F2 sfd_facts (the rendered fact card that F2 pushed into its prompt)
    columns    CSV header of the downloaded table

Arms (same BM25F code, k1=1.5, b=0.75 per field, idf over the per-query pool):
    F_tn    title + narrative       weights (w_title) tuned, w_narr = 1
    F_tnfc  title + narrative + facts + columns, (w_title, w_facts, w_cols) tuned
The single-field BM25 bugfix score (07_oracle_tokenizer.py) is the reference.

CV: the 25 queries are shuffled (seed 9) into 5 folds; for each fold the grid point
with the best mean NDCG@10 (pool all) on the other 20 queries is applied to the
held-out 5. Ties go to the first grid point (zero weights for the new fields first).
Both tokenizers from 07 are run.

Writes results/eval/bm25f.json and results/paired/bm25f.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import MANIFEST, RESULTS_DIR, TABLES_DIR, read_jsonl, select_benchmark  # noqa: E402
from tables import load_table  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ORACLE = _load("07_oracle_tokenizer")
EVAL, PAIRED = ORACLE.EVAL, ORACLE.PAIRED
FIELDS = ("title", "narrative", "facts", "columns")
K1, B = 1.5, 0.75
GRID_TITLE = (0.5, 1.0, 2.0, 3.0)
GRID_EXTRA = (0.0, 0.25, 0.5, 1.0, 2.0)


def build_fields(corpus, listed, sfd_arm: str, facts_arm: str) -> dict[str, dict[str, str]]:
    sfd = {r["dataset_id"]: r for r in read_jsonl(RESULTS_DIR / "descriptions" / f"{sfd_arm}.jsonl")}
    facts = {r["dataset_id"]: r for r in read_jsonl(RESULTS_DIR / "descriptions" / f"{facts_arm}.jsonl")}
    manifest = {r["dataset_id"]: r for r in read_jsonl(MANIFEST)}
    out = {}
    for d, entry in corpus.items():
        f = {"title": entry.get("title") or "", "narrative": entry.get("description") or "", "facts": "", "columns": ""}
        if d in listed:
            row = sfd.get(d, {})
            if row.get("status") == "ok" and row.get("sfd"):
                f["narrative"] = row["sfd"]
            frow = facts.get(d, {})
            if frow.get("status") == "ok":
                f["facts"] = frow.get("sfd_facts") or ""
            if manifest.get(d, {}).get("status") in {"ok", "truncated"}:
                df, _ = load_table(TABLES_DIR / f"{d}.csv", max_rows=5)
                f["columns"] = " ".join(str(c) for c in df.columns)
        out[d] = f
    return out


def bm25f_rank(tokenize, query: str, cand: list[str], fields: dict, weights: dict[str, float]) -> list[str]:
    toks = {d: {f: tokenize(fields[d][f]) for f in FIELDS} for d in cand}
    tfs = {d: {f: Counter(toks[d][f]) for f in FIELDS} for d in cand}
    avg = {f: (np.mean([len(toks[d][f]) for d in cand]) or 1.0) for f in FIELDS}
    n = len(cand)
    q = tokenize(query)
    scores = []
    for d in cand:
        s = 0.0
        for t in q:
            tf = sum(weights[f] * tfs[d][f][t] / (1 - B + B * len(toks[d][f]) / avg[f]) for f in FIELDS if weights[f])
            if tf <= 0:
                continue
            df = sum(1 for e in cand if any(weights[f] and tfs[e][f][t] for f in FIELDS))
            s += math.log(1 + (n - df + 0.5) / (df + 0.5)) * tf / (K1 + tf)
        scores.append(s)
    return [cand[i] for i in np.argsort(-np.array(scores), kind="stable")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--sfd-arm", default="bugfix__r1__subset30")
    ap.add_argument("--facts-arm", default="F2__r1__subset30")
    ap.add_argument("--dataset-ids", type=Path, default=HERE / "configs" / "subset30.txt")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=9)
    args = ap.parse_args()

    queries, qrels, corpus = select_benchmark(csv_only=True)
    manifest = {r["dataset_id"]: r for r in read_jsonl(MANIFEST)}
    with_table = {d for d in corpus if manifest.get(d, {}).get("status") in {"ok", "truncated"}}
    listed = {l.strip() for l in args.dataset_ids.read_text(encoding="utf-8").splitlines() if l.strip()} & set(corpus)
    subsets = {"all": set(corpus), "with_table": with_table, "listed": listed}
    fields = build_fields(corpus, listed, args.sfd_arm, args.facts_arm)
    qids = sorted(q for q in queries if any(r > 0 for r in qrels[q].values()))
    perm = np.random.default_rng(args.seed).permutation(len(qids))
    fold_of = {qids[i]: j % args.folds for j, i in enumerate(perm)}

    grids = {
        "F_tn": [dict(title=t, narrative=1.0, facts=0.0, columns=0.0) for t in GRID_TITLE],
        "F_tnfc": [dict(title=t, narrative=1.0, facts=fa, columns=c)
                   for fa, c, t in itertools.product(GRID_EXTRA, GRID_EXTRA, GRID_TITLE)],
    }

    per_query, chosen = {}, {}
    for tname, tok in ORACLE.TOKENIZERS.items():
        for arm, grid in grids.items():
            # NDCG@10 on pool "all" for every grid point and query (tuning objective)
            table = []
            for w in grid:
                row = {}
                for q in qids:
                    cand = sorted(qrels[q])
                    row[q] = EVAL.ndcg_at_k(bm25f_rank(tok, queries[q], cand, fields, w), qrels[q], 10)
                table.append(row)
            picks = {}
            for k in range(args.folds):
                train = [q for q in qids if fold_of[q] != k]
                best = max(range(len(grid)), key=lambda i: (np.mean([table[i][q] for q in train]), -i))
                picks[k] = best
            chosen[f"{arm}|{tname}"] = {k: grid[i] for k, i in picks.items()}
            for sname, allowed in subsets.items():
                scores = {}
                for q in qids:
                    judgments = {d: r for d, r in qrels[q].items() if d in allowed}
                    if not any(r > 0 for r in judgments.values()):
                        continue
                    ranked = bm25f_rank(tok, queries[q], sorted(judgments), fields, grid[picks[fold_of[q]]])
                    scores[q] = {f"ndcg@{k}": EVAL.ndcg_at_k(ranked, judgments, k) for k in (10, 20)}
                per_query[f"{arm}|{tname}|{sname}"] = scores
                means = " ".join(f"@{k}={np.mean([s[f'ndcg@{k}'] for s in scores.values()]):.4f}" for k in (10, 20))
                print(f"{arm}|{tname}|{sname:10s} n={len(scores):2d} {means}  picks={[grid[i] for i in picks.values()]}"
                      if sname == "all" else f"{arm}|{tname}|{sname:10s} n={len(scores):2d} {means}")

    ref = json.loads((RESULTS_DIR / "eval" / "oracle_tok.json").read_text(encoding="utf-8"))["per_query"]
    cmp_args = argparse.Namespace(bootstrap=10000, n_perm=10000)
    rows = []
    for tname in ORACLE.TOKENIZERS:
        for sname in subsets:
            pairs = [
                (f"F_tn|{tname}|{sname}", f"F_tnfc|{tname}|{sname}"),
                (f"bugfix__r1__subset30|{tname}|{sname}", f"F_tnfc|{tname}|{sname}"),
                (f"bugfix__r1__subset30|{tname}|{sname}", f"F_tn|{tname}|{sname}"),
            ]
            for a_key, b_key in pairs:
                sa = ref[a_key] if a_key in ref else per_query[a_key]
                for metric in ("ndcg@10", "ndcg@20"):
                    res = PAIRED.compare(sa, per_query[b_key], metric, cmp_args, np.random.default_rng(args.seed))
                    if res["n_queries"] == 0:
                        continue
                    res.update(a=a_key, b=b_key, metric=metric)
                    rows.append(res)
                    print(f"{b_key:28s} - {a_key:40s} {metric} n={res['n_queries']:2d} d={res['delta']:+.4f} "
                          f"W/T/L={res['wins']}/{res['ties']}/{res['losses']} p={res['p_randomization']:.4f} "
                          f"CI=[{res['ci95_low']:+.4f},{res['ci95_high']:+.4f}]")

    settings = {"k1": K1, "b": B, "fields": FIELDS, "grid_title": GRID_TITLE, "grid_extra": GRID_EXTRA,
                "folds": args.folds, "seed": args.seed, "tuning": "mean ndcg@10, pool all, train folds",
                "sfd_arm": args.sfd_arm, "facts_arm": args.facts_arm}
    (RESULTS_DIR / "eval" / "bm25f.json").write_text(json.dumps(
        {"settings": settings, "fold_of": fold_of, "chosen": chosen, "per_query": per_query}, indent=1), encoding="utf-8")
    (RESULTS_DIR / "paired" / "bm25f.json").write_text(json.dumps({"rows": rows}, indent=1), encoding="utf-8")
    print("wrote results/eval/bm25f.json results/paired/bm25f.json")


if __name__ == "__main__":
    main()
