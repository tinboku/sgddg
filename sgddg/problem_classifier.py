"""
Column problem classifier for routing columns to appropriate matching strategies.

Classifies each column by problem type (skip-KG, abbreviation, entity-linking,
domain-anchoring, semantic-search) before invoking KG matching.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple


class ProblemType(Enum):
    """Column problem type."""
    SKIP_KG = "skip_kg"
    ABBREVIATION = "abbreviation"
    ENTITY_LINKING = "entity_linking"
    DOMAIN_ANCHORING = "domain_anchoring"
    SEMANTIC_SEARCH = "semantic_search"


class MatchStrategy(Enum):
    """Matching strategy for a column."""
    NONE = "none"
    COLUMN_NAME = "column_name"
    COLUMN_AND_TYPE = "column_name+type"
    SAMPLE_VALUES = "sample_values"
    DATASET_CONTEXT = "dataset+column"


@dataclass
class ClassificationResult:
    """Classification output for a single column."""
    column_name: str
    problem_type: ProblemType
    match_strategy: MatchStrategy
    confidence: float
    reason: str
    should_query_kg: bool


# Patterns that indicate KG lookup is unnecessary
SKIP_KG_PATTERNS = [
    # ID columns
    (r'^id$', "Standard ID column"),
    (r'.*_id$', "Foreign key column"),
    (r'.*id$', "ID column (camelCase)"),

    # Timestamps
    (r'.*_at$', "Timestamp column"),
    (r'.*_date$', "Date column"),
    (r'.*_time$', "Time column"),
    (r'^created.*$', "Creation timestamp"),
    (r'^updated.*$', "Update timestamp"),
    (r'^deleted.*$', "Deletion flag"),
    (r'^timestamp$', "Timestamp column"),

    # Unique identifiers
    (r'^uuid$', "UUID column"),
    (r'^index$', "Index column"),
    (r'^row_?num(ber)?$', "Row number column"),
    (r'^key$', "Key column"),
    (r'^pk$', "Primary key column"),
    (r'^fk$', "Foreign key column"),

    # Boolean flags
    (r'^is_.*$', "Boolean flag column"),
    (r'^has_.*$', "Boolean flag column"),
    (r'^can_.*$', "Boolean flag column"),
    (r'^should_.*$', "Boolean flag column"),

    # Internal columns
    (r'^_.*$', "Internal/hidden column"),
    (r'^__.*$', "Private column"),

    # Unambiguous time parts
    (r'^year$', "Year column"),
    (r'^month$', "Month column"),
    (r'^day$', "Day column"),
    (r'^hour$', "Hour column"),
    (r'^minute$', "Minute column"),
    (r'^second$', "Second column"),
]

# Common abbreviations that need disambiguation
COMMON_ABBREVIATIONS = {
    'vol': ['volume', 'volatility'],
    'temp': ['temperature', 'temporary'],
    'qty': ['quantity'],
    'amt': ['amount'],
    'val': ['value', 'valid', 'validation'],
    'desc': ['description', 'descending'],
    'num': ['number'],
    'cnt': ['count'],
    'pct': ['percent', 'percentage'],
    'avg': ['average'],
    'min': ['minimum'],
    'max': ['maximum'],
    'dt': ['date', 'datetime'],
    'ts': ['timestamp'],
    'yr': ['year'],
    'mo': ['month'],
    'cat': ['category'],
    'src': ['source'],
    'dst': ['destination'],
    'txn': ['transaction'],
    'acc': ['account', 'accuracy'],
    'bal': ['balance'],
    'ref': ['reference'],
}

# Generic terms needing domain context
DOMAIN_ANCHORING_WORDS = {
    'price', 'rate', 'value', 'score', 'amount', 'level',
    'status', 'type', 'category', 'name', 'code', 'number',
    'date', 'time', 'count', 'total', 'average', 'percent',
}

# Standard unambiguous words
STANDARD_WORDS = {
    'id', 'name', 'date', 'time', 'year', 'month', 'day',
    'hour', 'minute', 'second', 'email', 'phone', 'address',
    'city', 'state', 'country', 'zip', 'latitude', 'longitude',
    'url', 'image', 'file', 'path', 'description', 'notes',
    'created', 'updated', 'deleted', 'active', 'enabled',
}


class ProblemClassifier:
    """Classifies columns by problem type to decide matching strategy."""

    def __init__(self, custom_abbreviations: Optional[Dict[str, List[str]]] = None):
        self.abbreviations = COMMON_ABBREVIATIONS.copy()
        if custom_abbreviations:
            self.abbreviations.update(custom_abbreviations)

    def classify(
        self,
        column_name: str,
        stats_profile: Optional[Dict[str, Any]] = None,
        dataset_name: Optional[str] = None
    ) -> ClassificationResult:
        """Classify a single column into a problem type."""
        col_lower = column_name.lower().strip()

        # Rule 1: Skip-KG patterns
        for pattern, reason in SKIP_KG_PATTERNS:
            if re.match(pattern, col_lower):
                return ClassificationResult(
                    column_name=column_name,
                    problem_type=ProblemType.SKIP_KG,
                    match_strategy=MatchStrategy.NONE,
                    confidence=0.95,
                    reason=reason,
                    should_query_kg=False
                )

        # Rule 2: Known abbreviations
        if col_lower in self.abbreviations:
            return ClassificationResult(
                column_name=column_name,
                problem_type=ProblemType.ABBREVIATION,
                match_strategy=MatchStrategy.COLUMN_AND_TYPE,
                confidence=0.85,
                reason=f"Abbreviation: may mean {self.abbreviations[col_lower]}",
                should_query_kg=True
            )

        # Rule 3: Short names (possible abbreviations)
        if len(column_name) <= 4 and col_lower not in STANDARD_WORDS:
            return ClassificationResult(
                column_name=column_name,
                problem_type=ProblemType.ABBREVIATION,
                match_strategy=MatchStrategy.COLUMN_AND_TYPE,
                confidence=0.70,
                reason=f"Short name, possibly abbreviation",
                should_query_kg=True
            )

        # Rule 4: High-cardinality text (entity linking candidate)
        if stats_profile:
            is_text = stats_profile.get('data_type') in ['string', 'object', 'str']
            unique_ratio = stats_profile.get('unique_ratio', 0)

            if is_text and unique_ratio > 0.8:
                samples = stats_profile.get('sample_values', [])
                if self._looks_like_entity_codes(samples):
                    return ClassificationResult(
                        column_name=column_name,
                        problem_type=ProblemType.ENTITY_LINKING,
                        match_strategy=MatchStrategy.SAMPLE_VALUES,
                        confidence=0.80,
                        reason="High-cardinality text with entity-like values",
                        should_query_kg=True
                    )

        # Rule 5: Exact domain-anchoring words
        if col_lower in DOMAIN_ANCHORING_WORDS:
            return ClassificationResult(
                column_name=column_name,
                problem_type=ProblemType.DOMAIN_ANCHORING,
                match_strategy=MatchStrategy.DATASET_CONTEXT,
                confidence=0.75,
                reason=f"Generic term '{col_lower}' needs domain context",
                should_query_kg=True
            )

        # Rule 6: Contains domain-anchoring words
        for word in DOMAIN_ANCHORING_WORDS:
            if word in col_lower:
                return ClassificationResult(
                    column_name=column_name,
                    problem_type=ProblemType.DOMAIN_ANCHORING,
                    match_strategy=MatchStrategy.DATASET_CONTEXT,
                    confidence=0.65,
                    reason=f"Contains generic term '{word}'",
                    should_query_kg=True
                )

        # Default: standard semantic search
        return ClassificationResult(
            column_name=column_name,
            problem_type=ProblemType.SEMANTIC_SEARCH,
            match_strategy=MatchStrategy.COLUMN_NAME,
            confidence=0.60,
            reason="Standard semantic search",
            should_query_kg=True
        )

    def classify_batch(
        self,
        columns: List[str],
        stats_profiles: Optional[List[Dict[str, Any]]] = None,
        dataset_name: Optional[str] = None
    ) -> List[ClassificationResult]:
        """Classify multiple columns at once."""
        results = []
        for i, col in enumerate(columns):
            profile = stats_profiles[i] if stats_profiles and i < len(stats_profiles) else None
            results.append(self.classify(col, profile, dataset_name))
        return results

    def get_skip_kg_columns(
        self,
        columns: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Partition columns into those needing KG and those that don't."""
        need_kg = []
        skip_kg = []

        for col in columns:
            result = self.classify(col)
            if result.should_query_kg:
                need_kg.append(col)
            else:
                skip_kg.append(col)

        return need_kg, skip_kg

    def _looks_like_entity_codes(self, samples: List[Any]) -> bool:
        """Check whether sample values resemble entity codes (e.g., stock tickers)."""
        if not samples:
            return False

        valid_count = 0
        for sample in samples[:10]:
            if not isinstance(sample, str):
                continue

            s = str(sample).strip()
            if not s:
                continue

            if (
                2 <= len(s) <= 6 and
                ' ' not in s and
                (s.isupper() or s.islower() or s.isdigit())
            ):
                valid_count += 1

        return valid_count >= len(samples) * 0.5

    def print_summary(self, results: List[ClassificationResult]) -> None:
        """Print a classification summary."""
        skip_count = sum(1 for r in results if not r.should_query_kg)
        query_count = len(results) - skip_count

        print(f"\nClassification Summary:")
        print(f"   Total columns: {len(results)}")
        print(f"   Skip KG: {skip_count} ({skip_count/len(results)*100:.1f}%)")
        print(f"   Query KG: {query_count} ({query_count/len(results)*100:.1f}%)")

        type_counts = {}
        for r in results:
            t = r.problem_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        print(f"\n   By Problem Type:")
        for t, count in sorted(type_counts.items()):
            print(f"     - {t}: {count}")


def classify_column(column_name: str, stats_profile: Optional[Dict] = None) -> ClassificationResult:
    """Convenience: classify a single column."""
    classifier = ProblemClassifier()
    return classifier.classify(column_name, stats_profile)


def should_query_kg(column_name: str) -> bool:
    """Convenience: check if a column needs KG lookup."""
    return classify_column(column_name).should_query_kg


if __name__ == "__main__":
    print("Testing Problem Classifier\n")

    classifier = ProblemClassifier()

    test_cases = [
        ("user_id", None, False),
        ("created_at", None, False),
        ("id", None, False),
        ("uuid", None, False),
        ("is_active", None, False),
        ("vol", None, True),
        ("temp", None, True),
        ("qty", None, True),
        ("amt", None, True),
        ("ticker", {"data_type": "string", "unique_ratio": 0.9, "sample_values": ["AAPL", "MSFT", "GOOG"]}, True),
        ("price", None, True),
        ("rate", None, True),
        ("total_value", None, True),
        ("company_name", None, True),
        ("product_category", None, True),
    ]

    print("=" * 70)
    print(f"{'Column':<20} {'Type':<18} {'Strategy':<18} {'Query KG':<10}")
    print("=" * 70)

    for col, profile, expected_query in test_cases:
        result = classifier.classify(col, profile)
        status = "PASS" if result.should_query_kg == expected_query else "FAIL"
        print(f"{col:<20} {result.problem_type.value:<18} {result.match_strategy.value:<18} {str(result.should_query_kg):<10} {status}")

    print("\n" + "=" * 70)

    all_columns = [t[0] for t in test_cases]
    results = classifier.classify_batch(all_columns)
    classifier.print_summary(results)
