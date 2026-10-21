"""Shared helpers for the AutoDDG-Deep experiment scripts.

Query/qrels/collection loading for NTCIR Data Search 2 (English), plus the
query and corpus filters used by every script so that all arms see the
same evaluation set.

Query set note: the "32 test queries" used throughout the plan are the test
topics with >= 3 relevant (L1/L2) judgments over the *full* collection
(min_relevant=3, min_total=1). AutoDDG's default filter (min_relevant=5,
min_total=10) keeps 23; re-filtering after a CSV-only restriction keeps 11-15.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_NTCIR_DIR = Path(
    os.path.expanduser(
        "~/Desktop/experiment11/code/knowledge_enhanced_metadata_generator/data/ntcir"
    )
)
DATA_DIR = HERE / "data"
TABLES_DIR = DATA_DIR / "tables"
MANIFEST = DATA_DIR / "manifest.jsonl"
CACHE_DIR = HERE / "cache"
RESULTS_DIR = HERE / "results"

RELEVANCE_MAP = {"L0": 0, "L1": 1, "L2": 2}


def load_topics(ntcir_dir: Path, split: str) -> dict[str, str]:
    topics = {}
    with open(ntcir_dir / f"data_search_2_e_{split}_topics.tsv", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                topics[parts[0].strip()] = parts[1].strip()
    return topics


def load_qrels(ntcir_dir: Path, split: str) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(ntcir_dir / f"data_search_2_e_{split}_qrels.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                qrels[parts[0]][parts[1]] = RELEVANCE_MAP.get(parts[-1], 0)
    return dict(qrels)


def load_collection(ntcir_dir: Path, ids: set[str] | None = None) -> dict[str, dict]:
    collection = {}
    with open(ntcir_dir / "data_search_e_collection.jsonl", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if ids is None or entry["id"] in ids:
                collection[entry["id"]] = entry
    return collection


def csv_urls(entry: dict) -> list[str]:
    """data_url of every file whose data_format is csv (same test as eval/benchmark_loader)."""
    return [
        d["data_url"]
        for d in entry.get("data", [])
        if (d.get("data_format") or "").lower().strip() == "csv" and d.get("data_url")
    ]


def filter_queries(
    topics: dict[str, str],
    qrels: dict[str, dict[str, int]],
    min_relevant: int,
    min_total: int,
) -> dict[str, str]:
    kept = {}
    for qid, text in topics.items():
        judgments = qrels.get(qid, {})
        if len(judgments) >= min_total and sum(r >= 1 for r in judgments.values()) >= min_relevant:
            kept[qid] = text
    return kept


def select_benchmark(
    ntcir_dir: Path = DEFAULT_NTCIR_DIR,
    split: str = "test",
    min_relevant: int = 3,
    min_total: int = 1,
    csv_only: bool = False,
    refilter_after_csv: bool = False,
) -> tuple[dict[str, str], dict[str, dict[str, int]], dict[str, dict]]:
    """Return (queries, qrels, corpus) where corpus is the judged pool of the kept queries.

    csv_only restricts the corpus (and qrels) to datasets with a CSV file.
    refilter_after_csv re-applies the query filter on the CSV-restricted qrels,
    as eval/benchmark_loader.py does; off by default so the query set stays fixed.
    """
    topics = load_topics(ntcir_dir, split)
    qrels = load_qrels(ntcir_dir, split)
    queries = filter_queries(topics, qrels, min_relevant, min_total)
    pool = {did for qid in queries for did in qrels[qid]}
    collection = load_collection(ntcir_dir, pool)
    if csv_only:
        collection = {did: e for did, e in collection.items() if csv_urls(e)}
        qrels = {
            qid: {did: r for did, r in qrels[qid].items() if did in collection} for qid in queries
        }
        if refilter_after_csv:
            queries = filter_queries(queries, qrels, min_relevant, min_total)
            pool = {did for qid in queries for did in qrels[qid]}
            collection = {did: e for did, e in collection.items() if did in pool}
    qrels = {qid: qrels[qid] for qid in queries}
    return queries, qrels, collection


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)
