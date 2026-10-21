"""Tokenizer fix and oracle ceiling for the subset30 BM25 evaluation (local, no API).

Usage:
    python -I 07_oracle_tokenizer.py

Step 1 (tokenizer): re-score existing arms with two BM25 tokenizers on the same
documents and queries:
    raw   text.lower().split()                           (03_index_eval.py)
    stem  lowercase, keep [a-z0-9]+ runs (punctuation removed), Porter stem
Step 2 (oracle, DIAGNOSTIC CEILING, USES TEST QRELS): for every subset30 table,
append the content words of the queries that judge it relevant (rel > 0) to its
bugfix SFD, deduplicated, at most --oracle-max-terms words, each word once.
This is not a deployable method; it measures how far the description channel
can move NDCG on this benchmark.

Also reports a ranking-level ceiling ("pool_ceiling"): the best NDCG@k reachable
from the bugfix ranking by re-placing only the subset30 documents (documents
outside subset30 keep their relative order; exact DP merge). No description of
the 30 tables can do better than this under per_query pooling and fixed BM25.

Settings follow 03_index_eval.py: BM25Okapi k1=1.5 b=0.75, per_query pool,
subsets all / with_table / listed, NDCG@10/20. Paired tests reuse 04_paired_test.

Writes results/eval/oracle_tok.json and results/paired/oracle_tok.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import MANIFEST, RESULTS_DIR, read_jsonl, select_benchmark  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EVAL = _load("03_index_eval")
PAIRED = _load("04_paired_test")

_STEMMER = PorterStemmer()
_STEM_CACHE: dict[str, str] = {}
_WORD = re.compile(r"[a-z0-9]+")
# function words dropped from oracle terms only (the tokenizers themselves keep everything)
STOPWORDS = set(
    "a an the of in on at to for and or by with from about into over under is are was were be been "
    "this that these those it its as per vs via what which who whom how when where why do does did "
    "data dataset datasets information list number".split()
)


def tok_raw(text: str) -> list[str]:
    return text.lower().split()


def _stem(w: str) -> str:
    s = _STEM_CACHE.get(w)
    if s is None:
        s = _STEM_CACHE[w] = _STEMMER.stem(w)
    return s


def tok_stem(text: str) -> list[str]:
    return [_stem(w) for w in _WORD.findall(text.lower())]


TOKENIZERS = {"raw": tok_raw, "stem": tok_stem}


def rank(tokenize, query: str, cand: list[str], docs: dict[str, str]) -> list[str]:
    bm25 = BM25Okapi([tokenize(docs[d]) for d in cand], k1=EVAL.BM25_K1, b=EVAL.BM25_B)
    scores = bm25.get_scores(tokenize(query))
    return [cand[i] for i in np.argsort(-scores, kind="stable")]


def oracle_terms(qids: list[str], queries: dict[str, str], max_terms: int) -> list[str]:
    seen, out = set(), []
    for q in qids:
        for w in _WORD.findall(queries[q].lower()):
            if w in STOPWORDS or w in seen:
                continue
            seen.add(w)
            out.append(w)
    return out[:max_terms]


def pool_ceiling(ranked: list[str], judgments: dict[str, int], listed: set[str], k: int) -> list[str]:
    """Best DCG@k ranking reachable by re-placing only the listed documents.

    Unlisted documents keep their relative order; listed ones are sorted by
    relevance (optimal among themselves) and merged in by DP over (i listed, j unlisted).
    """
    sub = sorted((d for d in ranked if d in listed), key=lambda d: -judgments.get(d, 0))
    rest = [d for d in ranked if d not in listed]
    gain = lambda d: 2 ** judgments.get(d, 0) - 1  # noqa: E731
    disc = lambda p: 1 / np.log2(p + 2) if p < k else 0.0  # noqa: E731
    best = np.full((len(sub) + 1, len(rest) + 1), -1.0)
    best[len(sub), len(rest)] = 0.0
    for i in range(len(sub), -1, -1):
        for j in range(len(rest), -1, -1):
            if i == len(sub) and j == len(rest):
                continue
            p = i + j
            opts = []
            if i < len(sub):
                opts.append(gain(sub[i]) * disc(p) + best[i + 1, j])
            if j < len(rest):
                opts.append(gain(rest[j]) * disc(p) + best[i, j + 1])
            best[i, j] = max(opts)
    out, i, j = [], 0, 0
    while i < len(sub) or j < len(rest):
        p = i + j
        take_sub = i < len(sub) and (
            j == len(rest) or gain(sub[i]) * disc(p) + best[i + 1, j] >= gain(rest[j]) * disc(p) + best[i, j + 1])
        if take_sub:
            out.append(sub[i]); i += 1  # noqa: E702
        else:
            out.append(rest[j]); j += 1  # noqa: E702
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--arms", default="baseline__r1__subset30,baseline__r2__subset30,bugfix__r1__subset30,F2__r1__subset30")
    ap.add_argument("--base-arm", default="bugfix__r1__subset30", help="SFD the oracle terms are appended to")
    ap.add_argument("--dataset-ids", type=Path, default=HERE / "configs" / "subset30.txt")
    ap.add_argument("--oracle-max-terms", type=int, default=40)
    ap.add_argument("--k", default="10,20")
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=9)
    args = ap.parse_args()

    ks = [int(k) for k in args.k.split(",")]
    queries, qrels, corpus = select_benchmark(csv_only=True)
    manifest = {r["dataset_id"]: r for r in read_jsonl(MANIFEST)}
    with_table = {d for d in corpus if manifest.get(d, {}).get("status") in {"ok", "truncated"}}
    listed = {l.strip() for l in args.dataset_ids.read_text(encoding="utf-8").splitlines() if l.strip()} & set(corpus)
    qids = sorted(queries)
    subsets = {"all": set(corpus), "with_table": with_table, "listed": listed}

    arm_docs = {arm: EVAL.arm_documents(arm, "sfd", corpus)[0] for arm in args.arms.split(",")}

    # oracle: DIAGNOSTIC CEILING built from test qrels
    base = arm_docs[args.base_arm]
    oracle_docs = dict(base)
    oracle_meta = {}
    for d in sorted(listed):
        rel_q = [q for q in qids if qrels[q].get(d, 0) > 0]
        terms = oracle_terms(rel_q, queries, args.oracle_max_terms)
        if terms:
            oracle_docs[d] = f"{base[d]} {' '.join(terms)}"
        oracle_meta[d] = {"relevant_queries": rel_q, "terms": terms}
    arm_docs["ORACLE_testqrels"] = oracle_docs

    per_query: dict[str, dict] = {}
    for tname, tok in TOKENIZERS.items():
        for sname, allowed in subsets.items():
            for arm, docs in arm_docs.items():
                scores, ceil = {}, {}
                for q in qids:
                    judgments = {d: r for d, r in qrels[q].items() if d in allowed}
                    if not any(r > 0 for r in judgments.values()):
                        continue
                    ranked = rank(tok, queries[q], sorted(judgments), docs)
                    scores[q] = {f"ndcg@{k}": EVAL.ndcg_at_k(ranked, judgments, k) for k in ks}
                    if arm == args.base_arm:
                        ceil[q] = {f"ndcg@{k}": EVAL.ndcg_at_k(pool_ceiling(ranked, judgments, listed, k), judgments, k)
                                   for k in ks}
                per_query[f"{arm}|{tname}|{sname}"] = scores
                if ceil:
                    per_query[f"POOL_CEILING|{tname}|{sname}"] = ceil

    for key, scores in per_query.items():
        means = " ".join(f"{m}={np.mean([s[m] for s in scores.values()]):.4f}" for m in (f"ndcg@{k}" for k in ks))
        print(f"{key:55s} n={len(scores):2d} {means}")

    # paired comparisons (B - A)
    cmp_args = argparse.Namespace(bootstrap=args.bootstrap, n_perm=args.n_perm)
    b = args.base_arm
    pairs = [(f"{b}|raw", f"{b}|stem")]  # tokenizer effect on the main arm
    for arm in arm_docs:
        if arm != b:
            pairs.append((f"{arm}|raw", f"{arm}|stem"))
    for t in TOKENIZERS:
        pairs += [
            (f"{b}|{t}", f"ORACLE_testqrels|{t}"),
            (f"{b}|{t}", f"POOL_CEILING|{t}"),
            ("baseline__r1__subset30|" + t, "baseline__r2__subset30|" + t),
            (f"{b}|{t}", f"F2__r1__subset30|{t}"),
        ]
    rows = []
    for a_key, b_key in pairs:
        for sname in subsets:
            for k in ks:
                metric = f"ndcg@{k}"
                res = PAIRED.compare(per_query[f"{a_key}|{sname}"], per_query[f"{b_key}|{sname}"], metric,
                                     cmp_args, np.random.default_rng(args.seed))
                if res["n_queries"] == 0:
                    continue
                res.update(a=a_key, b=b_key, subset=sname, metric=metric)
                rows.append(res)
                print(f"{b_key:35s} - {a_key:35s} {sname:10s} {metric} n={res['n_queries']:2d} "
                      f"d={res['delta']:+.4f} W/T/L={res['wins']}/{res['ties']}/{res['losses']} "
                      f"p={res['p_randomization']:.4f} CI=[{res['ci95_low']:+.4f},{res['ci95_high']:+.4f}]")

    settings = {
        "pool": "per_query", "bm25": {"k1": EVAL.BM25_K1, "b": EVAL.BM25_B},
        "tokenizers": {"raw": "lower().split()", "stem": "lower, [a-z0-9]+ runs, nltk PorterStemmer"},
        "oracle": {"WARNING": "diagnostic ceiling built from TEST qrels; not a deployable method",
                   "base_arm": b, "max_terms": args.oracle_max_terms, "stopwords_removed": sorted(STOPWORDS)},
        "n_queries": len(qids), "n_corpus": len(corpus), "n_listed": len(listed), "ks": ks,
        "n_perm": args.n_perm, "bootstrap": args.bootstrap, "seed": args.seed,
    }
    (RESULTS_DIR / "eval").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "paired").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "eval" / "oracle_tok.json").write_text(
        json.dumps({"settings": settings, "oracle_meta": oracle_meta, "per_query": per_query}, indent=1), encoding="utf-8")
    (RESULTS_DIR / "paired" / "oracle_tok.json").write_text(json.dumps({"rows": rows}, indent=1), encoding="utf-8")
    print("wrote results/eval/oracle_tok.json results/paired/oracle_tok.json")


if __name__ == "__main__":
    main()
