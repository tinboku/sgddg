"""Language Quality Evaluator - assesses readability, fluency, and information completeness."""

from typing import Dict, Any, List, Optional
import re


class LanguageQualityEvaluator:
    """Evaluates generated text quality across readability, fluency, and completeness."""

    @staticmethod
    def evaluate_readability(text: str) -> float:
        """Compute simplified Flesch Reading Ease score, normalized to 0-1."""
        if not text:
            return 0.0

        # Count sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = len(sentences)

        if num_sentences == 0:
            return 0.0

        # Count words
        words = text.split()
        num_words = len(words)

        if num_words == 0:
            return 0.0

        # Approximate syllable count
        num_syllables = int(num_words * 1.5)

        # Flesch Reading Ease formula
        score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)

        # Normalize to 0-1
        normalized = max(0, min(1, score / 100))

        return normalized

    @staticmethod
    def evaluate_fluency(text: str) -> float:
        """Evaluate fluency based on sentence length and vocabulary diversity, returning 0-1."""
        if not text:
            return 0.0

        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        # Sentence length variance
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_length = sum(sentence_lengths) / len(sentence_lengths)

        # Ideal sentence length: 15-25 words
        length_score = 1.0 if 15 <= avg_length <= 25 else 0.8

        # Vocabulary diversity (unique words / total words)
        words = text.lower().split()
        if len(words) == 0:
            return 0.0

        diversity = len(set(words)) / len(words)

        # Combined score
        fluency = (length_score + diversity) / 2

        return min(1.0, fluency)

    @staticmethod
    def evaluate_completeness(text: str, expected_elements: List[str]) -> float:
        """Evaluate information completeness by checking for expected keywords, returning 0-1."""
        if not text or not expected_elements:
            return 0.0

        text_lower = text.lower()

        # Count found elements
        found_count = sum(1 for elem in expected_elements if elem.lower() in text_lower)

        return found_count / len(expected_elements)

    @staticmethod
    def evaluate_text(
        text: str,
        expected_elements: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate overall text quality combining readability, fluency, and optionally completeness."""
        readability = LanguageQualityEvaluator.evaluate_readability(text)
        fluency = LanguageQualityEvaluator.evaluate_fluency(text)

        results = {
            "readability": readability,
            "fluency": fluency,
            "word_count": len(text.split()),
            "sentence_count": len(re.split(r'[.!?]+', text))
        }

        if expected_elements:
            completeness = LanguageQualityEvaluator.evaluate_completeness(text, expected_elements)
            results["completeness"] = completeness
            results["overall_score"] = (readability + fluency + completeness) / 3
        else:
            results["overall_score"] = (readability + fluency) / 2

        return results


if __name__ == "__main__":
    print("LanguageQualityEvaluator Demo")

    text = """
    This dataset provides a comprehensive analysis of financial performance.
    It includes key metrics such as revenue, profit, and operating margin.
    The data spans multiple years and can be used for trend analysis and forecasting.
    Researchers and analysts will find this dataset valuable for business intelligence purposes.
    """

    expected_elements = ["revenue", "profit", "margin", "analysis", "dataset"]

    print("\nEvaluating text quality...")
    results = LanguageQualityEvaluator.evaluate_text(text, expected_elements)

    print("\nResults:")
    print(f"   Readability: {results['readability']:.4f}")
    print(f"   Fluency: {results['fluency']:.4f}")
    print(f"   Completeness: {results['completeness']:.4f}")
    print(f"   Overall: {results['overall_score']:.4f}")
    print(f"   Word count: {results['word_count']}")
    print(f"   Sentence count: {results['sentence_count']}")

    print("\nDemo complete!")
