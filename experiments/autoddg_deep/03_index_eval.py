"""Index each arm's descriptions and score NDCG@k per query.

Usage:
    python -I 03_index_eval.py --arms baseline__r1,bugfix__r1 --k 10,20 --retrievers bm25,dense,hybrid
    python -I 03_index_eval.py --arms human --tag sanity          # title + original NTCIR description

An arm is a file results/descriptions/<arm>.jsonl from 02_generate.py, or the
built-in "human" arm (title + original description, no LLM). For every arm the
document text of a dataset is

    title + <view>        view in {ufd, sfd}, when 02_generate produced it
    title + description   otherwise (no table / generation error): the same
                          degraded input in every arm

Settings, identical across arms:
  bm25    rank_bm25.BM25Okapi, k1=1.5 b=0.75, tokens = text.lower().split()
          (as autoddg.ranking.metrics.downstream_task_rank)
  dense   fastembed BAAI/bge-small-en-v1.5, cosine (query_embed / passage_embed)
  hybrid  reciprocal rank fusion of bm25 and dense, k_rrf=60
  NDCG    gain 2^rel - 1, log2 discount, ideal = the query's judgments within the pool
  pool    per_query: rank only the query's judged datasets (AutoDDG NTCIR protocol)
          global:    rank the whole corpus
  subset  all | with_table (queries re-scored on datasets that have a usable table)

Writes results/eval/<tag>.json (per-query scores) and results/eval/<tag>_summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CACHE_DIR,
    DEFAULT_NTCIR_DIR,
    MANIFEST,
    RESULTS_DIR,
    read_jsonl,
    select_benchmark,
    sha256_text,
)

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
RRF_K = 60
BM25_K1, BM25_B = 1.5, 0.75


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def ndcg_at_k(ranked_ids: list[str], judgments: dict[str, int], k: int) -> float:
    gains = [2 ** judgments.get(d, 0) - 1 for d in ranked_ids[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted((2**r - 1 for r in judgments.values()), reverse=True)[:k]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else float("nan")


class DenseEncoder:
    """fastembed encoder with a per-text on-disk cache (same vectors for every arm)."""

    def __init__(self, model_name: str = DENSE_MODEL):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.model = TextEmbedding(model_name)
        self.cache_path = CACHE_DIR / "emb" / f"{model_name.replace('/', '__')}.npz"
        self.cache: dict[str, np.ndarray] = {}
        if self.cache_path.exists():
            with np.load(self.cache_path) as z:
                self.cache = {k: z[k] for k in z.files}

    def _encode(self, texts: list[str], kind: str) -> np.ndarray:
        keys = [f"{kind}_{sha256_text(t)}" for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if k not in self.cache]
        if missing:
            embed = self.model.query_embed if kind == "q" else self.model.passage_embed
            for (k, _), v in zip(missing, embed([t for _, t in missing])):
                self.cache[k] = np.asarray(v, dtype=np.float32)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.cache_path, **self.cache)
        m = np.stack([self.cache[k] for k in keys])
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    def queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, "q")

    def passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, "p")


def rank_bm25(query: str, doc_ids: list[str], docs: dict[str, str]) -> list[str]:
    bm25 = BM25Okapi([tokenize(docs[d]) for d in doc_ids], k1=BM25_K1, b=BM25_B)
    scores = bm25.get_scores(tokenize(query))
    order = np.argsort(-scores, kind="stable")
    return [doc_ids[i] for i in order]


def rank_dense(qvec: np.ndarray, doc_ids: list[str], dvecs: dict[str, np.ndarray]) -> list[str]:
    scores = np.array([float(dvecs[d] @ qvec) for d in doc_ids])
    order = np.argsort(-scores, kind="stable")
    return [doc_ids[i] for i in order]


def rrf(*rankings: list[str]) -> list[str]:
    score: dict[str, float] = {}
    for ranking in rankings:
        for r, d in enumerate(ranking):
            score[d] = score.get(d, 0.0) + 1.0 / (RRF_K + r + 1)
    return sorted(score, key=lambda d: (-score[d], d))


def arm_documents(arm: str, view: str, corpus: dict[str, dict]) -> tuple[dict[str, str], dict]:
    """Document text per dataset for one arm/view, plus coverage counts."""
    generated: dict[str, str] = {}
    if arm != "human":
        path = RESULTS_DIR / "descriptions" / f"{arm}.jsonl"
        if not path.exists():
            raise SystemExit(f"missing {path}; run 02_generate.py first")
        for row in read_jsonl(path):
            if row.get("status") == "ok" and row.get(view):
                generated[row["dataset_id"]] = row[view]
    docs, n_gen = {}, 0
    for did, entry in corpus.items():
        title = entry.get("title") or ""
        if did in generated:
            docs[did] = f"{title} {generated[did]}"
            n_gen += 1
        else:
            docs[did] = f"{title} {entry.get('description') or ''}"
    return docs, {"generated": n_gen, "degraded": len(corpus) - n_gen}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--arms", required=True, help="comma-separated, e.g. baseline__r1,bugfix__r1,human")
    ap.add_argument("--views", default="sfd,ufd", help="ignored for the human arm")
    ap.add_argument("--k", default="10,20")
    ap.add_argument("--retrievers", default="bm25,dense,hybrid")
    ap.add_argument("--pools", default="per_query,global")
    ap.add_argument("--subsets", default="all,with_table")
    ap.add_argument("--corpus", default="csv", choices=["csv", "all"],
                    help="csv: judged datasets with a CSV file; all: every judged dataset")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--ntcir-dir", type=Path, default=DEFAULT_NTCIR_DIR)
    ap.add_argument("--split", default="test")
    ap.add_argument("--min-relevant", type=int, default=3)
    ap.add_argument("--min-total", type=int, default=1)
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    ks = [int(k) for k in args.k.split(",")]
    retrievers = args.retrievers.split(",")
    pools = args.pools.split(",")
    subsets = args.subsets.split(",")
    tag = args.tag or "__".join(arms)

    queries, qrels, corpus = select_benchmark(
        args.ntcir_dir, args.split, args.min_relevant, args.min_total, csv_only=args.corpus == "csv"
    )
    manifest = {r["dataset_id"]: r for r in read_jsonl(MANIFEST)}
    with_table = {d for d in corpus if manifest.get(d, {}).get("status") in {"ok", "truncated"}}
    qids = sorted(queries)
    print(f"queries={len(qids)} corpus={len(corpus)} with_table={len(with_table)} corpus_mode={args.corpus}")

    encoder = DenseEncoder() if any(r in {"dense", "hybrid"} for r in retrievers) else None
    qvecs = dict(zip(qids, encoder.queries([queries[q] for q in qids]))) if encoder else {}

    per_query: dict[str, dict] = {}
    coverage: dict[str, dict] = {}
    for arm in arms:
        for view in (["title+description"] if arm == "human" else args.views.split(",")):
            docs, cov = arm_documents(arm, view, corpus)
            name = f"{arm}|{view}"
            coverage[name] = cov
            dvecs = {}
            if encoder:
                ids = sorted(docs)
                dvecs = dict(zip(ids, encoder.passages([docs[d] for d in ids])))
            for pool in pools:
                for subset in subsets:
                    allowed = with_table if subset == "with_table" else set(corpus)
                    global_ids = sorted(allowed)
                    for retriever in retrievers:
                        key = f"{name}|{retriever}|{pool}|{subset}"
                        scores = {}
                        for q in qids:
                            judgments = {d: r for d, r in qrels[q].items() if d in allowed}
                            if not any(r > 0 for r in judgments.values()):
                                continue  # no relevant dataset left in this pool
                            cand = sorted(judgments) if pool == "per_query" else global_ids
                            ranked = {}
                            if retriever in {"bm25", "hybrid"}:
                                ranked["bm25"] = rank_bm25(queries[q], cand, docs)
                            if retriever in {"dense", "hybrid"}:
                                ranked["dense"] = rank_dense(qvecs[q], cand, dvecs)
                            final = rrf(ranked["bm25"], ranked["dense"]) if retriever == "hybrid" else ranked[retriever]
                            scores[q] = {f"ndcg@{k}": ndcg_at_k(final, judgments, k) for k in ks}
                        per_query[key] = scores
                        means = {m: float(np.mean([s[m] for s in scores.values()])) for m in (f"ndcg@{k}" for k in ks)}
                        print(f"{key:70s} n_q={len(scores):2d} " + " ".join(f"{m}={v:.4f}" for m, v in means.items()))

    out_dir = RESULTS_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "split": args.split, "min_relevant": args.min_relevant, "min_total": args.min_total,
        "corpus": args.corpus, "n_queries": len(qids), "n_corpus": len(corpus),
        "n_with_table": len(with_table), "bm25": {"k1": BM25_K1, "b": BM25_B, "tokenizer": "lower().split()"},
        "dense_model": DENSE_MODEL, "rrf_k": RRF_K, "gain": "2^rel-1", "ks": ks,
    }
    (out_dir / f"{tag}.json").write_text(
        json.dumps({"settings": settings, "coverage": coverage, "per_query": per_query}, indent=1),
        encoding="utf-8",
    )
    with open(out_dir / f"{tag}_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "view", "retriever", "pool", "subset", "n_queries"] + [f"ndcg@{k}" for k in ks])
        for key, scores in per_query.items():
            w.writerow(key.split("|") + [len(scores)] + [
                round(float(np.mean([s[f"ndcg@{k}"] for s in scores.values()])), 4) for k in ks
            ])
    print(f"wrote {out_dir / (tag + '.json')}")


if __name__ == "__main__":
    main()
