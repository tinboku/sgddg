"""Tests for LanguageQualityEvaluator."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.language_quality import LanguageQualityEvaluator


def test_readability_nonempty():
    score = LanguageQualityEvaluator.evaluate_readability(
        "This dataset provides a comprehensive analysis of financial performance."
    )
    assert 0.0 <= score <= 1.0


def test_readability_empty():
    assert LanguageQualityEvaluator.evaluate_readability("") == 0.0


def test_fluency():
    score = LanguageQualityEvaluator.evaluate_fluency(
        "This dataset provides a comprehensive analysis of financial performance. "
        "It includes key metrics such as revenue, profit, and operating margin."
    )
    assert 0.0 <= score <= 1.0


def test_completeness():
    text = "This dataset includes revenue and profit."
    elements = ["revenue", "profit", "margin"]
    score = LanguageQualityEvaluator.evaluate_completeness(text, elements)
    assert abs(score - 2 / 3) < 0.01


def test_evaluate_text_with_expected():
    result = LanguageQualityEvaluator.evaluate_text(
        "Revenue and profit analysis.", expected_elements=["revenue", "profit"]
    )
    assert "readability" in result
    assert "overall_score" in result
    assert "completeness" in result


def test_evaluate_text_without_expected():
    result = LanguageQualityEvaluator.evaluate_text("Some text.")
    assert "readability" in result
    assert "overall_score" in result
    assert "completeness" not in result
