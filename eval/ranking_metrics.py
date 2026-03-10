"""Ranking Metrics - computes NDCG@K, Recall@K, MRR, MAP, and statistical significance tests."""

import numpy as np
from typing import List, Dict, Any, Tuple


class RankingMetrics:
    """Ranking evaluation metrics calculator."""

    @staticmethod
    def ndcg_at_k(relevance_scores: List[float], k: int) -> float:
        """Compute NDCG@K from a list of relevance scores ordered by retrieval rank."""
        if not relevance_scores or k <= 0:
            return 0.0

        # DCG@K
        dcg = sum(
            (2 ** rel - 1) / np.log2(i + 2)
            for i, rel in enumerate(relevance_scores[:k])
        )

        # IDCG@K (ideal ordering)
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = sum(
            (2 ** rel - 1) / np.log2(i + 2)
            for i, rel in enumerate(ideal_scores[:k])
        )

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def recall_at_k(relevant_items: set, retrieved_items: List[Any], k: int) -> float:
        """Compute Recall@K."""
        if not relevant_items:
            return 0.0

        retrieved_k = set(retrieved_items[:k])
        hits = len(relevant_items & retrieved_k)

        return hits / len(relevant_items)

    @staticmethod
    def precision_at_k(relevant_items: set, retrieved_items: List[Any], k: int) -> float:
        """Compute Precision@K."""
        if k <= 0:
            return 0.0

        retrieved_k = set(retrieved_items[:k])
        hits = len(relevant_items & retrieved_k)

        return hits / k

    @staticmethod
    def mrr(relevant_items: set, retrieved_items: List[Any]) -> float:
        """Compute Mean Reciprocal Rank (MRR)."""
        for i, item in enumerate(retrieved_items, start=1):
            if item in relevant_items:
                return 1.0 / i
        return 0.0

    @staticmethod
    def average_precision(relevant_items: set, retrieved_items: List[Any]) -> float:
        """Compute Average Precision (AP)."""
        if not relevant_items:
            return 0.0

        precisions = []
        hits = 0

        for i, item in enumerate(retrieved_items, start=1):
            if item in relevant_items:
                hits += 1
                precision_at_i = hits / i
                precisions.append(precision_at_i)

        return sum(precisions) / len(relevant_items) if precisions else 0.0

    @staticmethod
    def evaluate_ranking(
        query_results: List[Tuple[str, List[str], set]],
        k_values: List[int] = [1, 5, 10, 20]
    ) -> Dict[str, Any]:
        """Batch evaluate ranking results with binary relevance."""
        metrics_results = {
            "ndcg": {k: [] for k in k_values},
            "recall": {k: [] for k in k_values},
            "precision": {k: [] for k in k_values},
            "mrr": [],
            "map": []
        }

        for query, retrieved, relevant in query_results:
            relevance_scores = [1.0 if item in relevant else 0.0 for item in retrieved]

            for k in k_values:
                ndcg = RankingMetrics.ndcg_at_k(relevance_scores, k)
                metrics_results["ndcg"][k].append(ndcg)

            for k in k_values:
                recall = RankingMetrics.recall_at_k(relevant, retrieved, k)
                metrics_results["recall"][k].append(recall)

            for k in k_values:
                precision = RankingMetrics.precision_at_k(relevant, retrieved, k)
                metrics_results["precision"][k].append(precision)

            mrr = RankingMetrics.mrr(relevant, retrieved)
            metrics_results["mrr"].append(mrr)

            ap = RankingMetrics.average_precision(relevant, retrieved)
            metrics_results["map"].append(ap)

        # Compute averages
        summary = {
            "ndcg": {k: np.mean(scores) for k, scores in metrics_results["ndcg"].items()},
            "recall": {k: np.mean(scores) for k, scores in metrics_results["recall"].items()},
            "precision": {k: np.mean(scores) for k, scores in metrics_results["precision"].items()},
            "mrr": np.mean(metrics_results["mrr"]),
            "map": np.mean(metrics_results["map"])
        }

        return {
            "summary": summary,
            "detailed": metrics_results,
            "num_queries": len(query_results)
        }

    @staticmethod
    def evaluate_ranking_graded(
        query_results: List[Tuple[str, List[str], Dict[str, int]]],
        k_values: List[int] = [1, 5, 10, 20],
        dcg_mode: str = "exponential",
    ) -> Dict[str, Any]:
        """Batch evaluate ranking results with graded relevance (L0/L1/L2)."""
        metrics_results = {
            "ndcg": {k: [] for k in k_values},
            "recall": {k: [] for k in k_values},
            "precision": {k: [] for k in k_values},
            "mrr": [],
            "map": [],
        }

        for query_id, retrieved, qrel_dict in query_results:
            relevance_scores = [float(qrel_dict.get(item, 0)) for item in retrieved]

            all_relevance = list(qrel_dict.values()) + [0.0] * max(0, len(retrieved) - len(qrel_dict))

            for k in k_values:
                full_retrieved_rel = relevance_scores[:k]
                ideal_scores = sorted(all_relevance, reverse=True)

                if dcg_mode == "linear":
                    dcg = sum(
                        rel / np.log2(i + 2)
                        for i, rel in enumerate(full_retrieved_rel)
                    )
                    idcg = sum(
                        rel / np.log2(i + 2)
                        for i, rel in enumerate(ideal_scores[:k])
                    )
                else:
                    dcg = sum(
                        (2 ** rel - 1) / np.log2(i + 2)
                        for i, rel in enumerate(full_retrieved_rel)
                    )
                    idcg = sum(
                        (2 ** rel - 1) / np.log2(i + 2)
                        for i, rel in enumerate(ideal_scores[:k])
                    )
                ndcg_val = dcg / idcg if idcg > 0 else 0.0
                metrics_results["ndcg"][k].append(ndcg_val)

            # Binary threshold for Recall/Precision/MRR/MAP (L1+)
            relevant_set = {did for did, rel in qrel_dict.items() if rel >= 1}

            for k in k_values:
                recall = RankingMetrics.recall_at_k(relevant_set, retrieved, k)
                metrics_results["recall"][k].append(recall)

                precision = RankingMetrics.precision_at_k(relevant_set, retrieved, k)
                metrics_results["precision"][k].append(precision)

            mrr = RankingMetrics.mrr(relevant_set, retrieved)
            metrics_results["mrr"].append(mrr)

            ap = RankingMetrics.average_precision(relevant_set, retrieved)
            metrics_results["map"].append(ap)

        # Compute averages
        summary = {
            "ndcg": {k: float(np.mean(scores)) if scores else 0.0
                     for k, scores in metrics_results["ndcg"].items()},
            "recall": {k: float(np.mean(scores)) if scores else 0.0
                       for k, scores in metrics_results["recall"].items()},
            "precision": {k: float(np.mean(scores)) if scores else 0.0
                          for k, scores in metrics_results["precision"].items()},
            "mrr": float(np.mean(metrics_results["mrr"])) if metrics_results["mrr"] else 0.0,
            "map": float(np.mean(metrics_results["map"])) if metrics_results["map"] else 0.0,
        }

        return {
            "summary": summary,
            "detailed": metrics_results,
            "num_queries": len(query_results),
        }


    @staticmethod
    def significance_test(
        scores_a: List[float],
        scores_b: List[float],
        test: str = "wilcoxon",
    ) -> Dict[str, Any]:
        """Run statistical significance test between two sets of per-query scores."""
        from scipy import stats

        n = min(len(scores_a), len(scores_b))
        if n < 3:
            return {
                "statistic": 0.0, "p_value": 1.0,
                "significant_005": False, "significant_001": False,
                "mean_a": 0.0, "mean_b": 0.0, "mean_diff": 0.0, "n": n,
            }

        a = np.array(scores_a[:n])
        b = np.array(scores_b[:n])

        if test == "wilcoxon":
            diffs = a - b
            if np.all(diffs == 0):
                stat, p_val = 0.0, 1.0
            else:
                stat, p_val = stats.wilcoxon(a, b)
        elif test == "ttest":
            stat, p_val = stats.ttest_rel(a, b)
        else:
            raise ValueError(f"Unknown test: {test}")

        return {
            "statistic": float(stat),
            "p_value": float(p_val),
            "significant_005": p_val < 0.05,
            "significant_001": p_val < 0.01,
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "mean_diff": float(np.mean(b - a)),
            "n": n,
        }

    @staticmethod
    def pairwise_significance(
        method_scores: Dict[str, List[float]],
        test: str = "wilcoxon",
    ) -> Dict[str, Dict[str, Any]]:
        """Run pairwise significance tests between all method pairs."""
        methods = list(method_scores.keys())
        results = {}

        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                key = f"{methods[i]}_vs_{methods[j]}"
                results[key] = RankingMetrics.significance_test(
                    method_scores[methods[i]],
                    method_scores[methods[j]],
                    test=test,
                )

        return results


if __name__ == "__main__":
    print("RankingMetrics Demo")

    # Example: 3 queries
    query_results = [
        ("revenue", ["rev1", "inc1", "rev2", "sale1"], {"rev1", "rev2"}),
        ("profit", ["prof1", "earn1", "inc2"], {"prof1"}),
        ("age", ["age1", "age2", "year1"], {"age1", "age2"})
    ]

    print("\nEvaluating ranking results...")
    results = RankingMetrics.evaluate_ranking(query_results, k_values=[1, 3, 5])

    print("\nSummary:")
    summary = results["summary"]

    print("\n  NDCG@K:")
    for k, score in summary["ndcg"].items():
        print(f"    NDCG@{k}: {score:.4f}")

    print("\n  Recall@K:")
    for k, score in summary["recall"].items():
        print(f"    Recall@{k}: {score:.4f}")

    print(f"\n  MRR: {summary['mrr']:.4f}")
    print(f"  MAP: {summary['map']:.4f}")

    print(f"\n  Queries: {results['num_queries']}")

    print("\nDemo complete!")
