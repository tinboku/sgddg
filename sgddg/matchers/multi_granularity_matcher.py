"""
Multi-granularity matcher for SGDDG.

Fuses signals from column name, sample values, statistics patterns,
dataset context, and external KG matches to improve matching accuracy.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum


class MatchSignal(Enum):
    """Types of matching signals."""
    COLUMN_NAME = "column_name"
    SAMPLE_VALUES = "sample_values"
    STATS_PATTERN = "stats_pattern"
    DATASET_CONTEXT = "dataset_context"
    KG_MATCH = "kg_match"


@dataclass
class SignalMatch:
    """A single signal's match result."""
    signal_type: MatchSignal
    concept: str
    score: float
    confidence: float
    evidence: str


@dataclass
class FusedMatch:
    """Result after fusing multiple signal matches."""
    concept: str
    final_score: float
    semantic_strength: str  # EXACT, RELATED, CONTEXTUAL
    signals: List[SignalMatch]
    weights_used: Dict[str, float]
    reason: str


# Signal weight configurations per problem type
WEIGHT_CONFIGS = {
    "entity_linking": {
        "column_name": 0.05,
        "sample_values": 0.45,
        "stats_pattern": 0.05,
        "dataset_context": 0.05,
        "kg_match": 0.4,
    },
    "abbreviation": {
        "column_name": 0.3,
        "sample_values": 0.1,
        "stats_pattern": 0.1,
        "dataset_context": 0.1,
        "kg_match": 0.4,
    },
    "domain_anchoring": {
        "column_name": 0.15,
        "sample_values": 0.05,
        "stats_pattern": 0.1,
        "dataset_context": 0.3,
        "kg_match": 0.4,
    },
    "semantic_search": {
        "column_name": 0.1,
        "sample_values": 0.1,
        "stats_pattern": 0.1,
        "dataset_context": 0.1,
        "kg_match": 0.6,
    },
    "default": {
        "column_name": 0.1,
        "sample_values": 0.1,
        "stats_pattern": 0.1,
        "dataset_context": 0.1,
        "kg_match": 0.6,
    },
}


class MultiGranularityMatcher:
    """Fuses multiple signal sources to produce more accurate KG matches."""

    def __init__(self, weight_configs: Optional[Dict] = None):
        self.weight_configs = weight_configs or WEIGHT_CONFIGS

    def get_weights(self, problem_type: str) -> Dict[str, float]:
        """Get signal weights for the given problem type."""
        return self.weight_configs.get(problem_type, self.weight_configs["default"])

    def fuse_signals(
        self,
        signal_matches: Dict[MatchSignal, List[SignalMatch]],
        problem_type: str = "default"
    ) -> List[FusedMatch]:
        """Fuse matches from multiple signals into ranked results."""
        weights = self.get_weights(problem_type)

        # Collect all candidate concepts
        concept_scores: Dict[str, Dict] = {}

        for signal_type, matches in signal_matches.items():
            signal_key = signal_type.value
            weight = weights.get(signal_key, 0.1)

            for match in matches:
                concept = match.concept.lower()

                if concept not in concept_scores:
                    concept_scores[concept] = {
                        "total_score": 0.0,
                        "signals": [],
                        "weights_sum": 0.0,
                    }

                weighted_score = match.score * weight * match.confidence
                concept_scores[concept]["total_score"] += weighted_score
                concept_scores[concept]["weights_sum"] += weight
                concept_scores[concept]["signals"].append(match)

        # Normalize and build fused results
        fused_results = []

        for concept, data in concept_scores.items():
            if data["weights_sum"] > 0:
                final_score = data["total_score"] / data["weights_sum"]
            else:
                final_score = data["total_score"]

            semantic_strength = self._determine_semantic_strength(
                final_score,
                len(data["signals"])
            )

            reason = self._generate_reason(data["signals"], weights)

            fused_results.append(FusedMatch(
                concept=concept,
                final_score=final_score,
                semantic_strength=semantic_strength,
                signals=data["signals"],
                weights_used=weights,
                reason=reason
            ))

        fused_results.sort(key=lambda x: x.final_score, reverse=True)
        return fused_results

    def _determine_semantic_strength(
        self,
        score: float,
        signal_count: int
    ) -> str:
        """Classify match strength based on score and signal diversity."""
        if score >= 0.85 and signal_count >= 2:
            return "EXACT"
        elif score >= 0.6:
            return "RELATED"
        else:
            return "CONTEXTUAL"

    def _generate_reason(
        self,
        signals: List[SignalMatch],
        weights: Dict[str, float]
    ) -> str:
        """Generate a human-readable explanation of the match."""
        signal_types = [s.signal_type.value for s in signals]
        unique_types = list(set(signal_types))

        if len(unique_types) >= 3:
            return f"Multi-signal match: {', '.join(unique_types)}"
        elif len(unique_types) == 2:
            return f"Cross-validated: {' + '.join(unique_types)}"
        elif len(unique_types) == 1:
            return f"Single signal: {unique_types[0]}"
        else:
            return "No signal"

    def match_column(
        self,
        column_name: str,
        sample_values: List[Any],
        stats_profile: Dict[str, Any],
        dataset_name: Optional[str] = None,
        problem_type: str = "default",
        kg_match: Optional[Dict[str, Any]] = None
    ) -> List[FusedMatch]:
        """Run multi-granularity matching on a single column."""
        # Build augmented query for retrieval guidance
        dtype = stats_profile.get('data_type', 'unknown')
        unique_ratio = stats_profile.get('unique_ratio', 1.0)
        query_hint = f"Column: {column_name}, Type: {dtype}, Unique: {unique_ratio:.2f}"
        if dataset_name:
            query_hint += f", Dataset: {dataset_name}"

        print(f"    Augmented Query for Retrieval: [{query_hint}]")

        signal_matches = {}

        # 1. External KG match signal
        if kg_match and kg_match.get("status") == "matched":
            concept_name = kg_match.get("concept", {}).get("display_name", "")
            score = kg_match.get("score", 0.0)
            signal_matches[MatchSignal.KG_MATCH] = [SignalMatch(
                signal_type=MatchSignal.KG_MATCH,
                concept=concept_name,
                score=score,
                confidence=1.0,
                evidence=f"Augmented KG Match ({query_hint})"
            )]

        # 2. Sample values signal
        if sample_values:
            value_matches = self._match_by_sample_values(sample_values)
            if value_matches:
                signal_matches[MatchSignal.SAMPLE_VALUES] = value_matches

        # 3. Stats pattern signal
        pattern_matches = self._match_by_stats_pattern(stats_profile)
        if pattern_matches:
            signal_matches[MatchSignal.STATS_PATTERN] = pattern_matches

        # 4. Dataset context signal
        if dataset_name:
            context_matches = self._match_by_dataset_context(
                column_name, dataset_name
            )
            if context_matches:
                signal_matches[MatchSignal.DATASET_CONTEXT] = context_matches

        return self.fuse_signals(signal_matches, problem_type)

    def _match_by_column_name(self, column_name: str) -> List[SignalMatch]:
        """Basic column-name matching (can be overridden by subclasses)."""
        normalized = column_name.lower().replace("_", " ")

        return [SignalMatch(
            signal_type=MatchSignal.COLUMN_NAME,
            concept=normalized,
            score=1.0,
            confidence=0.8,
            evidence=f"Column name: {column_name}"
        )]

    def _match_by_sample_values(
        self,
        sample_values: List[Any]
    ) -> List[SignalMatch]:
        """Match based on sample value patterns (regex + heuristics)."""
        if not sample_values:
            return []

        matches = []
        samples_str = [str(v) for v in sample_values if v is not None]
        if not samples_str:
            return []

        avg_len = sum(len(s) for s in samples_str) / len(samples_str)

        # Ticker symbol detection (2-6 uppercase letters)
        if 2 <= avg_len <= 6 and all(s.isupper() and s.isalpha() for s in samples_str[:10]):
            matches.append(SignalMatch(
                signal_type=MatchSignal.SAMPLE_VALUES,
                concept="Ticker Symbol",
                score=0.85,
                confidence=0.8,
                evidence=f"Values look like tickers (avg_len={avg_len:.1f})"
            ))

        # Regex pattern matching
        import re
        regex_patterns = {
            "Email Address": r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$",
            "URL": r"^https?://[\w\.-]+",
            "IPv4 Address": r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$",
            "ISO Currency Code": r"^[A-Z]{3}$",
            "Date String": r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$"
        }

        for concept, pattern in regex_patterns.items():
            match_count = sum(1 for s in samples_str[:20] if re.match(pattern, s))
            if match_count / min(len(samples_str), 20) > 0.7:
                matches.append(SignalMatch(
                    signal_type=MatchSignal.SAMPLE_VALUES,
                    concept=concept,
                    score=0.9,
                    confidence=0.9,
                    evidence=f"Majority of samples match {concept} pattern"
                ))

        return matches

    def _match_by_stats_pattern(
        self,
        stats_profile: Dict[str, Any]
    ) -> List[SignalMatch]:
        """Match based on statistical distribution patterns."""
        matches = []

        data_type = stats_profile.get("data_type", "unknown")
        unique_ratio = stats_profile.get("unique_ratio", 0)
        null_rate = stats_profile.get("null_rate", 0)

        # High uniqueness + text type => likely an identifier
        if unique_ratio > 0.9 and "object" in str(data_type):
            matches.append(SignalMatch(
                signal_type=MatchSignal.STATS_PATTERN,
                concept="identifier",
                score=0.7,
                confidence=0.7,
                evidence=f"High unique ratio ({unique_ratio:.2f}) suggests identifier"
            ))

        # Low uniqueness => likely a category
        if unique_ratio < 0.1 and unique_ratio > 0:
            matches.append(SignalMatch(
                signal_type=MatchSignal.STATS_PATTERN,
                concept="category",
                score=0.7,
                confidence=0.7,
                evidence=f"Low unique ratio ({unique_ratio:.2f}) suggests category"
            ))

        return matches

    def _match_by_dataset_context(
        self,
        column_name: str,
        dataset_name: str
    ) -> List[SignalMatch]:
        """Match based on dataset-level domain context."""
        matches = []

        dataset_lower = dataset_name.lower()

        if "financial" in dataset_lower or "esg" in dataset_lower:
            matches.append(SignalMatch(
                signal_type=MatchSignal.DATASET_CONTEXT,
                concept="financial metric",
                score=0.6,
                confidence=0.6,
                evidence=f"Dataset '{dataset_name}' suggests financial domain"
            ))

        if "covid" in dataset_lower or "hospital" in dataset_lower:
            matches.append(SignalMatch(
                signal_type=MatchSignal.DATASET_CONTEXT,
                concept="healthcare metric",
                score=0.6,
                confidence=0.6,
                evidence=f"Dataset '{dataset_name}' suggests healthcare domain"
            ))

        return matches

    def print_match_results(self, results: List[FusedMatch], top_n: int = 3):
        """Print top fused match results."""
        print(f"\nTop {min(top_n, len(results))} Matches:")
        print("-" * 60)

        for i, match in enumerate(results[:top_n]):
            strength_icon = {
                "EXACT": "[EXACT]",
                "RELATED": "[REL]",
                "CONTEXTUAL": "[CTX]"
            }.get(match.semantic_strength, "[?]")

            print(f"{i+1}. [{strength_icon} {match.semantic_strength}] {match.concept}")
            print(f"   Score: {match.final_score:.2f}")
            print(f"   Signals: {len(match.signals)}")
            print(f"   Reason: {match.reason}")
            print()


def fuse_kg_matches(
    column_name: str,
    sample_values: List[Any],
    stats_profile: Dict[str, Any],
    problem_type: str = "default"
) -> List[FusedMatch]:
    """Convenience function to run multi-granularity matching."""
    matcher = MultiGranularityMatcher()
    return matcher.match_column(
        column_name, sample_values, stats_profile,
        problem_type=problem_type
    )


if __name__ == "__main__":
    print("Testing Multi-Granularity Matcher\n")

    matcher = MultiGranularityMatcher()

    print("=" * 60)
    print("Test Case 1: Stock Ticker Column")
    print("=" * 60)

    results = matcher.match_column(
        column_name="ticker",
        sample_values=["AAPL", "MSFT", "GOOG", "AMZN"],
        stats_profile={
            "data_type": "object",
            "unique_ratio": 0.95,
            "null_rate": 0.0
        },
        dataset_name="Stock Market Data",
        problem_type="entity_linking"
    )
    matcher.print_match_results(results)

    print("=" * 60)
    print("Test Case 2: ESG Score Column")
    print("=" * 60)

    results = matcher.match_column(
        column_name="ESG_Score",
        sample_values=[75.5, 82.3, 68.9, 91.2],
        stats_profile={
            "data_type": "float64",
            "unique_ratio": 0.8,
            "null_rate": 0.05
        },
        dataset_name="Company ESG Financial Dataset",
        problem_type="domain_anchoring"
    )
    matcher.print_match_results(results)

    print("=" * 60)
    print("Test Case 3: Abbreviation Column")
    print("=" * 60)

    results = matcher.match_column(
        column_name="vol",
        sample_values=[1000000, 2500000, 850000],
        stats_profile={
            "data_type": "int64",
            "unique_ratio": 0.95,
            "null_rate": 0.0
        },
        dataset_name="Trading Data",
        problem_type="abbreviation"
    )
    matcher.print_match_results(results)
