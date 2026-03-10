"""Tests for RankingMetrics."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.ranking_metrics import RankingMetrics


def test_ndcg_at_k_perfect():
    """All relevant items at the top should yield high NDCG."""
    relevance = [1.0, 1.0, 0.0, 0.0]
    score = RankingMetrics.ndcg_at_k(relevance, k=4)
    assert score > 0.9


def test_ndcg_at_k_worst():
    """Relevant item at the bottom should yield low NDCG."""
    relevance = [0.0, 0.0, 0.0, 1.0]
    score = RankingMetrics.ndcg_at_k(relevance, k=4)
    assert score < 0.5


def test_ndcg_at_k_empty():
    assert RankingMetrics.ndcg_at_k([], k=5) == 0.0


def test_precision_at_k():
    relevant = {"a", "c"}
    retrieved = ["a", "b", "c", "d"]
    score = RankingMetrics.precision_at_k(relevant, retrieved, k=4)
    assert score == 0.5


def test_recall_at_k():
    relevant = {"a", "b", "c"}
    retrieved = ["a", "d", "c"]
    score = RankingMetrics.recall_at_k(relevant, retrieved, k=3)
    assert abs(score - 2 / 3) < 0.01


def test_mrr():
    relevant = {"c"}
    retrieved = ["a", "b", "c"]
    score = RankingMetrics.mrr(relevant, retrieved)
    assert abs(score - 1 / 3) < 0.01


def test_average_precision():
    relevant = {"a", "c"}
    retrieved = ["a", "b", "c", "d"]
    ap = RankingMetrics.average_precision(relevant, retrieved)
    # AP = (1/1 + 2/3) / 2 = 0.833...
    assert abs(ap - (1.0 + 2 / 3) / 2) < 0.01
