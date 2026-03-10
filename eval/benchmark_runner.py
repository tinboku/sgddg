"""Benchmark Runner - end-to-end evaluation pipeline reporting NDCG, MAP, MRR on DDG benchmarks."""

import os
import sys
import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from eval.ranking_metrics import RankingMetrics
from eval.benchmark_loader import BenchmarkData
from eval.bm25_retriever import BM25Retriever, create_bm25_retriever


class StandardBenchmarkRunner:
    """Standard DDG benchmark evaluation pipeline following AutoDDG methodology."""

    def __init__(
        self,
        benchmark_data: BenchmarkData,
        output_dir: str = "results",
        k_values: Optional[List[int]] = None,
    ):
        """
        Args:
            benchmark_data: Loaded benchmark (NTCIR or ECIR)
            output_dir: Directory to save evaluation results
            k_values: K values for NDCG@K, Recall@K, etc.
        """
        self.benchmark = benchmark_data
        self.output_dir = output_dir
        self.k_values = k_values or [1, 5, 10, 20]
        os.makedirs(output_dir, exist_ok=True)

    def _build_index_text(
        self,
        dataset_id: str,
        descriptions: Optional[Dict[str, Dict[str, str]]],
        index_fields: List[str],
    ) -> str:
        """
        Build the text to index for a dataset.

        Args:
            dataset_id: Dataset identifier
            descriptions: {dataset_id: {"ufd": ..., "sfd": ...}}
            index_fields: Which fields to include, e.g. ["title", "ufd"] or ["title", "sfd"]
        """
        parts = []
        dataset_info = self.benchmark.datasets.get(dataset_id, {})

        for field in index_fields:
            if field == "title":
                title = (
                    dataset_info.get("title")
                    or dataset_info.get("name")
                    or dataset_info.get("dataset_name")
                    or ""
                )
                if title:
                    parts.append(title)

            elif field == "description":
                desc = (
                    dataset_info.get("description")
                    or dataset_info.get("notes")
                    or dataset_info.get("content")
                    or dataset_info.get("text")
                    or ""
                )
                if desc:
                    parts.append(desc)

            elif field == "columns":
                columns = dataset_info.get("columns") or dataset_info.get("schema") or []
                if isinstance(columns, list):
                    col_text = " ".join(str(c) for c in columns)
                elif isinstance(columns, dict):
                    col_text = " ".join(columns.keys())
                else:
                    col_text = ""
                if col_text.strip():
                    parts.append(col_text)

            elif field in ("ufd", "sfd"):
                if descriptions and dataset_id in descriptions:
                    text = descriptions[dataset_id].get(field, "")
                    if text:
                        parts.append(text)

        return " ".join(parts)

    def _get_all_dataset_ids_in_qrels(self) -> set:
        """Get all dataset IDs referenced in qrels."""
        all_ids = set()
        for qrel_dict in self.benchmark.qrels.values():
            all_ids.update(qrel_dict.keys())
        return all_ids

    def run_experiment(
        self,
        experiment_name: str,
        descriptions: Optional[Dict[str, Dict[str, str]]] = None,
        index_fields: List[str] = None,
        k_values: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Run a single experiment: build index -> retrieve -> evaluate.

        Args:
            experiment_name: Name for this experiment
            descriptions: {dataset_id: {"ufd": ..., "sfd": ...}}
            index_fields: Fields to index, e.g. ["title", "ufd"]
                         Defaults to ["title", "description"] if no descriptions provided
            k_values: Override k_values for this experiment

        Returns:
            Evaluation results dict
        """
        if index_fields is None:
            if descriptions:
                index_fields = ["title", "ufd"]
            else:
                index_fields = ["title", "description"]

        k_vals = k_values or self.k_values
        print(f"\n--- Experiment: {experiment_name} ---")
        print(f"  Index fields: {index_fields}")

        # Step 1: Build documents for indexing
        # Include all datasets that appear in qrels + collection
        all_qrel_ids = self._get_all_dataset_ids_in_qrels()
        all_collection_ids = set(self.benchmark.datasets.keys())

        # Use union: index everything we can
        dataset_ids_to_index = all_qrel_ids | all_collection_ids
        if descriptions:
            dataset_ids_to_index |= set(descriptions.keys())

        documents = {}
        for did in dataset_ids_to_index:
            text = self._build_index_text(did, descriptions, index_fields)
            if text.strip():
                documents[did] = text

        print(f"  Indexed {len(documents)} documents")

        # Step 2: Build BM25 index
        retriever = create_bm25_retriever(k1=1.5, b=0.75)
        retriever.index(documents)

        # Step 3: Retrieve for all queries
        retrieved_results = retriever.batch_search(
            self.benchmark.queries,
            top_k=max(k_vals) * 5,  # retrieve more to cover all k values
        )
        print(f"  Retrieved results for {len(retrieved_results)} queries")

        # Step 4: Evaluate
        is_graded = self.benchmark.stats.get("relevance_type") == "graded"

        if is_graded:
            # Use graded evaluation
            query_results = []
            for qid in self.benchmark.queries:
                if qid not in self.benchmark.qrels:
                    continue
                retrieved = retrieved_results.get(qid, [])
                qrel_dict = self.benchmark.qrels[qid]
                query_results.append((qid, retrieved, qrel_dict))

            eval_result = RankingMetrics.evaluate_ranking_graded(
                query_results, k_vals
            )
        else:
            # Use binary evaluation
            query_results = []
            for qid in self.benchmark.queries:
                if qid not in self.benchmark.qrels:
                    continue
                retrieved = retrieved_results.get(qid, [])
                relevant = {
                    did for did, rel in self.benchmark.qrels[qid].items() if rel >= 1
                }
                query_results.append((qid, retrieved, relevant))

            eval_result = RankingMetrics.evaluate_ranking(
                query_results, k_vals
            )

        # Add experiment metadata
        eval_result["experiment_name"] = experiment_name
        eval_result["index_fields"] = index_fields
        eval_result["num_indexed_docs"] = len(documents)
        eval_result["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # Print summary
        summary = eval_result.get("summary", {})
        ndcg = summary.get("ndcg", {})
        print(f"  Results:")
        for k in k_vals:
            print(f"    NDCG@{k}: {ndcg.get(k, 0.0):.4f}")
        print(f"    MAP: {summary.get('map', 0.0):.4f}")
        print(f"    MRR: {summary.get('mrr', 0.0):.4f}")

        return eval_result

    def run_full_evaluation(
        self,
        generator_adapter=None,
        methods: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete evaluation matrix.

        Args:
            generator_adapter: DescriptionGeneratorAdapter instance
            methods: List of methods to evaluate. Defaults to all.

        Returns:
            Full evaluation results with comparison table
        """
        if methods is None:
            methods = ["original", "title_only", "columns_only"]
            if generator_adapter:
                methods.extend(["sgddg_ufd", "sgddg_sfd", "no_kg_ufd", "no_kg_sfd"])

        results = {}
        all_descriptions = {}

        print(f"\n{'='*80}")
        print(f"FULL EVALUATION: {self.benchmark.name}")
        print(f"  Queries: {len(self.benchmark.queries)}")
        print(f"  Datasets: {len(self.benchmark.datasets)}")
        print(f"  Methods: {methods}")
        print(f"{'='*80}")

        # Generate descriptions for methods that need them
        if generator_adapter:
            if any(m.startswith("sgddg") for m in methods):
                print("\n[Generating sgDDG descriptions...]")
                all_descriptions["sgddg"] = generator_adapter.generate_all(
                    self.benchmark.datasets,
                    method="sgddg",
                    benchmark_name=self.benchmark.name,
                )

            if any(m.startswith("no_kg") for m in methods):
                print("\n[Generating no-KG descriptions...]")
                all_descriptions["no_kg"] = generator_adapter.generate_all(
                    self.benchmark.datasets,
                    method="no_kg",
                    benchmark_name=self.benchmark.name,
                )

            if "columns_only" in methods:
                print("\n[Generating columns-only descriptions...]")
                all_descriptions["columns_only"] = generator_adapter.generate_all(
                    self.benchmark.datasets,
                    method="columns_only",
                    benchmark_name=self.benchmark.name,
                )

        # Run experiments
        for method in methods:
            if method == "original":
                results[method] = self.run_experiment(
                    experiment_name="Original (title+desc)",
                    descriptions=None,
                    index_fields=["title", "description"],
                )
            elif method == "title_only":
                results[method] = self.run_experiment(
                    experiment_name="Title-only baseline",
                    descriptions=None,
                    index_fields=["title"],
                )
            elif method == "columns_only":
                results[method] = self.run_experiment(
                    experiment_name="Title + Column names",
                    descriptions=all_descriptions.get("columns_only"),
                    index_fields=["title", "ufd"],
                )
            elif method == "sgddg_ufd":
                results[method] = self.run_experiment(
                    experiment_name="sgDDG UFD",
                    descriptions=all_descriptions.get("sgddg"),
                    index_fields=["title", "ufd"],
                )
            elif method == "sgddg_sfd":
                results[method] = self.run_experiment(
                    experiment_name="sgDDG SFD",
                    descriptions=all_descriptions.get("sgddg"),
                    index_fields=["title", "sfd"],
                )
            elif method == "no_kg_ufd":
                results[method] = self.run_experiment(
                    experiment_name="sgDDG-noKG UFD",
                    descriptions=all_descriptions.get("no_kg"),
                    index_fields=["title", "ufd"],
                )
            elif method == "no_kg_sfd":
                results[method] = self.run_experiment(
                    experiment_name="sgDDG-noKG SFD",
                    descriptions=all_descriptions.get("no_kg"),
                    index_fields=["title", "sfd"],
                )

        # Build comparison table
        comparison = self._build_comparison_table(results)

        full_results = {
            "benchmark": self.benchmark.name,
            "benchmark_stats": self.benchmark.stats,
            "experiments": results,
            "comparison": comparison,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        return full_results

    def _build_comparison_table(
        self, results: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build a comparison table across experiments."""
        table = []

        # Add AutoDDG reference rows (NTCIR)
        if "ntcir" in self.benchmark.name.lower():
            table.append({
                "method": "AutoDDG-GPT SFD (paper)",
                "NDCG@20": 0.725,
                "NDCG@10": None,
                "NDCG@5": None,
                "MAP": None,
                "MRR": None,
                "num_queries": 32,
                "is_reference": True,
            })
            table.append({
                "method": "AutoDDG-GPT UFD (paper)",
                "NDCG@20": 0.670,
                "NDCG@10": None,
                "NDCG@5": None,
                "MAP": None,
                "MRR": None,
                "num_queries": 32,
                "is_reference": True,
            })

        for method_key, result in results.items():
            summary = result.get("summary", {})
            ndcg = summary.get("ndcg", {})
            row = {
                "method": result.get("experiment_name", method_key),
                "num_queries": result.get("num_queries", 0),
                "is_reference": False,
            }
            for k in self.k_values:
                row[f"NDCG@{k}"] = round(ndcg.get(k, 0.0), 4)
            row["MAP"] = round(summary.get("map", 0.0), 4)
            row["MRR"] = round(summary.get("mrr", 0.0), 4)
            table.append(row)

        return table

    def print_comparison_table(self, results: Dict[str, Any]):
        """Print a formatted comparison table."""
        comparison = results.get("comparison", [])
        if not comparison:
            return

        print(f"\n{'='*100}")
        print(f"BENCHMARK COMPARISON: {results.get('benchmark', 'N/A')}")
        stats = results.get("benchmark_stats", {})
        print(f"  Queries: {stats.get('num_queries', '?')}, "
              f"Datasets: {stats.get('num_datasets', '?')}, "
              f"Relevance: {stats.get('relevance_type', '?')}")
        print(f"{'='*100}")

        # Header
        header_parts = ["Method".ljust(30)]
        for k in self.k_values:
            header_parts.append(f"NDCG@{k}".rjust(9))
        header_parts.extend(["MAP".rjust(9), "MRR".rjust(9), "Queries".rjust(8)])
        print(" | ".join(header_parts))
        print("-" * 100)

        # Rows
        for row in comparison:
            is_ref = row.get("is_reference", False)
            name = row["method"]
            if is_ref:
                name = f"* {name}"

            parts = [name.ljust(30)]
            for k in self.k_values:
                val = row.get(f"NDCG@{k}")
                if val is not None:
                    parts.append(f"{val:.4f}".rjust(9))
                else:
                    parts.append("-".rjust(9))

            map_val = row.get("MAP")
            mrr_val = row.get("MRR")
            parts.append(f"{map_val:.4f}".rjust(9) if map_val is not None else "-".rjust(9))
            parts.append(f"{mrr_val:.4f}".rjust(9) if mrr_val is not None else "-".rjust(9))
            parts.append(str(row.get("num_queries", "-")).rjust(8))

            print(" | ".join(parts))

        print("=" * 100)
        print("  * = reference values from published paper (not reproduced)")

    def save_results(
        self, results: Dict[str, Any], filename: Optional[str] = None
    ):
        """Save evaluation results to JSON file."""
        if filename is None:
            filename = f"benchmark_{self.benchmark.name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        output_path = os.path.join(self.output_dir, filename)

        # Convert numpy types to Python native types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=convert)

        print(f"\nResults saved to: {output_path}")
        return output_path

    def print_results(self, results: Dict[str, Any]):
        """Print results (single or comparison)."""
        if "comparison" in results:
            self.print_comparison_table(results)
        else:
            # Single experiment
            summary = results.get("summary", {})
            print(f"\n{'='*60}")
            print(f"Experiment: {results.get('experiment_name', 'N/A')}")
            print(f"Queries: {results.get('num_queries', 0)}")
            print("-" * 60)
            ndcg = summary.get("ndcg", {})
            for k in self.k_values:
                print(f"  NDCG@{k}: {ndcg.get(k, 0.0):.4f}")
            print(f"  MAP:     {summary.get('map', 0.0):.4f}")
            print(f"  MRR:     {summary.get('mrr', 0.0):.4f}")
            print("=" * 60)


# Keep backward-compatible BenchmarkRunner for legacy code
class BenchmarkRunner(StandardBenchmarkRunner):
    """Legacy interface (backward compatible)."""

    def __init__(
        self,
        benchmark_dir: str = "",
        output_dir: str = "eval_results",
        k_values: Optional[List[int]] = None,
    ):
        # Create a minimal BenchmarkData for backward compat
        from eval.benchmark_loader import BenchmarkLoader
        self.benchmark_dir = benchmark_dir
        self.output_dir = output_dir
        self.k_values = k_values or [1, 5, 10, 20]
        self.loader = BenchmarkLoader(benchmark_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Placeholder benchmark data
        self.benchmark = BenchmarkData(
            name="legacy",
            queries={},
            qrels={},
            datasets={},
        )

    def evaluate_retrieval(
        self,
        retrieved_results: Dict[str, List[str]],
        qrels: Dict[str, Dict[str, int]],
        experiment_name: str = "default",
    ) -> Dict[str, Any]:
        """Backward-compatible evaluate_retrieval."""
        query_results = []
        for query_id, retrieved_ids in retrieved_results.items():
            if query_id not in qrels:
                continue
            relevant = self.loader.get_relevant_datasets(qrels, query_id)
            query_results.append((query_id, retrieved_ids, relevant))

        if not query_results:
            return {"error": "No overlapping queries"}

        eval_result = RankingMetrics.evaluate_ranking(query_results, self.k_values)
        eval_result["experiment_name"] = experiment_name
        eval_result["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        return eval_result

    def run_ablation_matrix(
        self,
        experiments: Dict[str, Dict[str, List[str]]],
        qrels: Dict[str, Dict[str, int]],
    ) -> Dict[str, Any]:
        """Backward-compatible ablation matrix."""
        results = {}
        for exp_name, retrieved in experiments.items():
            result = self.evaluate_retrieval(retrieved, qrels, exp_name)
            results[exp_name] = result

        comparison = self._build_comparison_table(results)
        return {"experiments": results, "comparison": comparison}
