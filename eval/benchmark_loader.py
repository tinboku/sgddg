"""Benchmark Loader - loads NTCIR-DDG (graded) and ECIR-DDG (binary) benchmark datasets."""

import os
import json
import csv
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path


@dataclass
class BenchmarkData:
    """Unified data container for DDG benchmarks."""
    name: str
    queries: Dict[str, str]  # query_id -> query_text
    qrels: Dict[str, Dict[str, int]]  # query_id -> {dataset_id: relevance}
    datasets: Dict[str, Any]  # dataset_id -> dataset_info dict
    stats: Dict[str, Any] = field(default_factory=dict)

    def get_relevant_datasets(self, query_id: str, min_relevance: int = 1) -> Set[str]:
        """Get dataset IDs with relevance >= min_relevance for a query."""
        if query_id not in self.qrels:
            return set()
        return {
            did for did, rel in self.qrels[query_id].items()
            if rel >= min_relevance
        }

    def get_graded_relevance(self, query_id: str) -> Dict[str, int]:
        """Get the full graded relevance dict for a query."""
        return self.qrels.get(query_id, {})

    def summary(self) -> str:
        """Return a human-readable summary of the benchmark."""
        total_judgments = sum(len(v) for v in self.qrels.values())
        lines = [
            f"Benchmark: {self.name}",
            f"  Queries: {len(self.queries)}",
            f"  Datasets: {len(self.datasets)}",
            f"  Judgments: {total_judgments}",
        ]
        if self.stats:
            for k, v in self.stats.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


class NTCIRLoader:
    """
    Load real NTCIR-DDG data (graded relevance L0/L1/L2).
    Source: HuggingFace mpkato/ntcir_data_search
    """

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            data_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "ntcir"
            )
        self.data_dir = data_dir

    def load(
        self,
        split: str = "test",
        min_relevant: int = 5,
        min_total: int = 10,
        auto_download: bool = True,
        judged_only: bool = False,
        csv_only: bool = False,
    ) -> BenchmarkData:
        """
        Load NTCIR-DDG benchmark data.

        Args:
            split: "test" or "train"
            min_relevant: Min relevant datasets per query (AutoDDG default: 5)
            min_total: Min total judged datasets per query (AutoDDG default: 10)
            auto_download: Automatically download if data not found
            judged_only: Only include datasets that appear in qrels (judged pool)
            csv_only: Only include datasets with CSV format data files

        Returns:
            BenchmarkData with graded relevance
        """
        # Check if data exists, download if needed
        topics_file = os.path.join(self.data_dir, f"data_search_2_e_{split}_topics.tsv")
        qrels_file = os.path.join(self.data_dir, f"data_search_2_e_{split}_qrels.txt")
        collection_file = os.path.join(self.data_dir, "data_search_e_collection.jsonl")
        collection_bz2 = os.path.join(self.data_dir, "data_search_e_collection.jsonl.bz2")

        if not os.path.exists(topics_file) or not os.path.exists(qrels_file):
            if auto_download:
                print("NTCIR data not found locally, downloading from HuggingFace...")
                from eval.download_ntcir import download_and_prepare
                result = download_and_prepare(
                    data_dir=self.data_dir,
                    split=split,
                    min_relevant=min_relevant,
                    min_total=min_total,
                )
                return BenchmarkData(
                    name=f"ntcir_ddg_{split}",
                    queries=result["queries"],
                    qrels=result["qrels"],
                    datasets=result["collection"],
                    stats=result["stats"],
                )
            else:
                raise FileNotFoundError(
                    f"NTCIR data not found at {self.data_dir}. "
                    f"Run: python -m eval.download_ntcir --data_dir {self.data_dir}"
                )

        # Parse topics
        queries = self._parse_topics(topics_file)

        # Parse qrels
        qrels = self._parse_qrels(qrels_file)

        # Apply AutoDDG-style query filtering
        queries, qrels = self._filter_queries(queries, qrels, min_relevant, min_total)

        # Load collection
        collection = {}
        if os.path.exists(collection_file):
            collection = self._load_collection(collection_file)
        elif os.path.exists(collection_bz2):
            # Decompress first
            from eval.download_ntcir import decompress_collection
            jsonl_path = decompress_collection(collection_bz2)
            collection = self._load_collection(jsonl_path)

        # Apply CSV-only filter: keep only datasets that have CSV format data files
        if csv_only:
            csv_ids = set()
            for did, info in collection.items():
                for d in info.get("data", []):
                    fmt = d.get("data_format", "").lower().strip()
                    if fmt == "csv":
                        csv_ids.add(did)
                        break
            collection = {did: info for did, info in collection.items() if did in csv_ids}
            # Also filter qrels to only include CSV datasets
            for qid in list(qrels.keys()):
                qrels[qid] = {did: rel for did, rel in qrels[qid].items() if did in csv_ids}
            # Re-filter queries after CSV restriction
            queries, qrels = self._filter_queries(queries, qrels, min_relevant, min_total)

        # Apply judged-only filter: keep only datasets that appear in any query's qrels
        if judged_only:
            judged_ids = set()
            for qrel_dict in qrels.values():
                judged_ids.update(qrel_dict.keys())
            collection = {did: info for did, info in collection.items() if did in judged_ids}

        # Compute stats
        total_judgments = sum(len(v) for v in qrels.values())
        l2 = sum(1 for q in qrels.values() for r in q.values() if r == 2)
        l1 = sum(1 for q in qrels.values() for r in q.values() if r == 1)
        l0 = sum(1 for q in qrels.values() for r in q.values() if r == 0)

        filter_mode = "full"
        if csv_only and judged_only:
            filter_mode = "csv_judged"
        elif csv_only:
            filter_mode = "csv_only"
        elif judged_only:
            filter_mode = "judged_only"

        stats = {
            "num_queries": len(queries),
            "num_datasets": len(collection),
            "num_judgments": total_judgments,
            "relevance_distribution": {"L0": l0, "L1": l1, "L2": l2},
            "relevance_type": "graded",
            "split": split,
            "filter_mode": filter_mode,
        }

        return BenchmarkData(
            name=f"ntcir_ddg_{split}",
            queries=queries,
            qrels=qrels,
            datasets=collection,
            stats=stats,
        )

    def _parse_topics(self, filepath: str) -> Dict[str, str]:
        """Parse NTCIR topics TSV: QUERY_ID<tab>QUERY_TEXT"""
        queries = {}
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    queries[parts[0].strip()] = parts[1].strip()
        return queries

    def _parse_qrels(self, filepath: str) -> Dict[str, Dict[str, int]]:
        """Parse NTCIR qrels: QUERY_ID DOC_ID L{0,1,2}"""
        from eval.download_ntcir import RELEVANCE_MAP

        qrels: Dict[str, Dict[str, int]] = {}
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    qid = parts[0]
                    doc_id = parts[1]
                    rel_label = parts[-1]

                    if rel_label in RELEVANCE_MAP:
                        rel = RELEVANCE_MAP[rel_label]
                    else:
                        try:
                            rel = int(rel_label)
                        except ValueError:
                            continue

                    if qid not in qrels:
                        qrels[qid] = {}
                    qrels[qid][doc_id] = rel

        return qrels

    def _filter_queries(
        self,
        queries: Dict[str, str],
        qrels: Dict[str, Dict[str, int]],
        min_relevant: int,
        min_total: int,
    ) -> Tuple[Dict[str, str], Dict[str, Dict[str, int]]]:
        """Filter queries following AutoDDG methodology."""
        filtered_q = {}
        filtered_r = {}
        for qid, text in queries.items():
            if qid not in qrels:
                continue
            judgments = qrels[qid]
            if len(judgments) >= min_total:
                relevant = sum(1 for r in judgments.values() if r >= 1)
                if relevant >= min_relevant:
                    filtered_q[qid] = text
                    filtered_r[qid] = judgments
        return filtered_q, filtered_r

    def _load_collection(self, jsonl_path: str) -> Dict[str, Dict[str, Any]]:
        """Load collection from JSONL."""
        collection = {}
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    doc_id = str(
                        entry.get("id")
                        or entry.get("doc_id")
                        or entry.get("dataset_id")
                        or entry.get("_id")
                        or ""
                    )
                    if doc_id:
                        collection[doc_id] = entry
                except json.JSONDecodeError:
                    continue
        return collection


class ECIRLoader:
    """
    Load ECIR-DDG data (binary relevance).
    Format: CSV with topic,relevance,document columns.
    """

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            data_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "ecir"
            )
        self.data_dir = data_dir

    def load(self) -> Optional[BenchmarkData]:
        """
        Load ECIR-DDG benchmark data.

        Returns:
            BenchmarkData with binary relevance, or None if data not found
        """
        # Try multiple possible file layouts
        queries_file = None
        qrels_file = None
        collection_dir = None

        # Layout 1: queries.json + qrels.csv
        for qf in ["queries.json", "queries.tsv", "topics.tsv"]:
            p = os.path.join(self.data_dir, qf)
            if os.path.exists(p):
                queries_file = p
                break

        for rf in ["qrels.csv", "qrels.txt", "qrels.tsv", "relevance.csv"]:
            p = os.path.join(self.data_dir, rf)
            if os.path.exists(p):
                qrels_file = p
                break

        for cd in ["datasets", "collection", "documents"]:
            p = os.path.join(self.data_dir, cd)
            if os.path.isdir(p):
                collection_dir = p
                break

        if not queries_file or not qrels_file:
            return None

        # Parse queries
        queries = self._parse_queries(queries_file)

        # Parse qrels
        qrels = self._parse_qrels(qrels_file)

        # Load collection metadata
        datasets = {}
        if collection_dir:
            datasets = self._load_datasets(collection_dir)

        total_judgments = sum(len(v) for v in qrels.values())
        stats = {
            "num_queries": len(queries),
            "num_datasets": len(datasets),
            "num_judgments": total_judgments,
            "relevance_type": "binary",
        }

        return BenchmarkData(
            name="ecir_ddg",
            queries=queries,
            qrels=qrels,
            datasets=datasets,
            stats=stats,
        )

    def _parse_queries(self, filepath: str) -> Dict[str, str]:
        """Parse queries from JSON or TSV."""
        queries = {}
        if filepath.endswith(".json"):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for entry in data:
                        qid = str(entry.get("query_id", entry.get("id", "")))
                        text = entry.get("query_text", entry.get("query", ""))
                        if qid and text:
                            queries[qid] = text
                elif isinstance(data, dict):
                    queries = {str(k): str(v) for k, v in data.items()}
        else:
            # TSV format
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t", 1)
                    if len(parts) == 2:
                        queries[parts[0].strip()] = parts[1].strip()
        return queries

    def _parse_qrels(self, filepath: str) -> Dict[str, Dict[str, int]]:
        """Parse ECIR qrels from CSV: topic,relevance,document"""
        qrels: Dict[str, Dict[str, int]] = {}

        with open(filepath, "r", encoding="utf-8") as f:
            # Try CSV with header
            sample = f.read(1024)
            f.seek(0)

            if "," in sample:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) >= 3:
                        # topic, relevance, document
                        qid = row[0].strip()
                        try:
                            rel = int(row[1].strip())
                        except ValueError:
                            rel = 1 if row[1].strip().lower() in ("true", "yes", "relevant") else 0
                        doc_id = row[2].strip()

                        if qid not in qrels:
                            qrels[qid] = {}
                        qrels[qid][doc_id] = rel
            else:
                # Space or tab separated: query_id doc_id relevance
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        qid = parts[0]
                        doc_id = parts[1]
                        try:
                            rel = int(parts[-1])
                        except ValueError:
                            continue
                        if qid not in qrels:
                            qrels[qid] = {}
                        qrels[qid][doc_id] = rel

        return qrels

    def _load_datasets(self, datasets_dir: str) -> Dict[str, Dict[str, Any]]:
        """Load dataset metadata from a directory."""
        datasets = {}
        for fname in os.listdir(datasets_dir):
            if fname.endswith((".json", ".csv", ".jsonl")):
                dataset_id = os.path.splitext(fname)[0]
                fpath = os.path.join(datasets_dir, fname)
                try:
                    if fname.endswith(".json"):
                        with open(fpath, "r", encoding="utf-8") as f:
                            datasets[dataset_id] = json.load(f)
                    elif fname.endswith(".csv"):
                        datasets[dataset_id] = {
                            "id": dataset_id,
                            "path": fpath,
                            "title": dataset_id.replace("_", " ").title(),
                        }
                except Exception:
                    continue
        return datasets


def load_benchmark(
    benchmark: str = "ntcir",
    data_dir: Optional[str] = None,
    **kwargs,
) -> BenchmarkData:
    """
    Convenience function to load a benchmark by name.

    Args:
        benchmark: "ntcir" or "ecir"
        data_dir: Override default data directory
        **kwargs: Passed to the loader (split, min_relevant, min_total, etc.)

    Returns:
        BenchmarkData
    """
    if benchmark.lower() in ("ntcir", "ntcir_ddg", "ntcir-ddg"):
        loader = NTCIRLoader(data_dir=data_dir)
        return loader.load(**kwargs)
    elif benchmark.lower() in ("ecir", "ecir_ddg", "ecir-ddg"):
        loader = ECIRLoader(data_dir=data_dir)
        result = loader.load()
        if result is None:
            raise FileNotFoundError(
                f"ECIR data not found at {data_dir or 'default location'}. "
                "ECIR data must be obtained separately."
            )
        return result
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}. Use 'ntcir' or 'ecir'.")


# Keep backward-compatible BenchmarkLoader for legacy code
class BenchmarkLoader:
    """Legacy loader interface (backward compatible). Prefer NTCIRLoader/ECIRLoader directly."""

    def __init__(self, benchmark_dir: str):
        self.benchmark_dir = Path(benchmark_dir)

    def load_ntcir_ddg(self) -> Optional[BenchmarkData]:
        ntcir_dir = self.benchmark_dir / "ntcir"
        if ntcir_dir.exists():
            loader = NTCIRLoader(data_dir=str(ntcir_dir))
            try:
                return loader.load(auto_download=False)
            except FileNotFoundError:
                return None
        return None

    def load_ecir_ddg(self) -> Optional[BenchmarkData]:
        ecir_dir = self.benchmark_dir / "ecir"
        if ecir_dir.exists():
            loader = ECIRLoader(data_dir=str(ecir_dir))
            return loader.load()
        return None

    def get_relevant_datasets(
        self, qrels: Dict[str, Dict[str, int]], query_id: str, min_relevance: int = 1
    ) -> set:
        """Get the set of relevant dataset IDs for a query."""
        if query_id not in qrels:
            return set()
        return {did for did, rel in qrels[query_id].items() if rel >= min_relevance}

    def list_available_benchmarks(self) -> List[str]:
        available = []
        for name in ["ntcir", "ecir"]:
            path = self.benchmark_dir / name
            if path.exists():
                available.append(name)
        return available
