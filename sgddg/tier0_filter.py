"""
Tier 0 filter for SGDDG.

Filters out common-sense concepts that LLMs already understand (e.g., "Year",
"Country") to avoid wasting prompt tokens on low-value KG injections.
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import json
import os


class ConceptTier(Enum):
    """Knowledge tier classification."""
    TIER_0 = "tier_0"  # Common sense (LLM already knows)
    TIER_1 = "tier_1"  # Domain terminology (valuable to inject)
    TIER_2 = "tier_2"  # Named entities (high value)
    TIER_3 = "tier_3"  # Cross-table relationships (highest value)
    UNKNOWN = "unknown"


@dataclass
class FilterResult:
    """Filtering decision for a single concept."""
    concept_name: str
    tier: ConceptTier
    should_inject: bool
    weight: float  # Injection weight (0.0 - 1.0)
    reason: str


# Tier 0 blacklist: concepts LLMs already understand well
TIER_0_BLACKLIST: Set[str] = {
    # Time concepts
    "year", "month", "day", "date", "time", "hour", "minute", "second",
    "week", "quarter", "timestamp", "datetime", "period",

    # Basic identifiers
    "id", "name", "title", "label", "code", "key", "index", "number",
    "identifier", "uuid", "guid",

    # Geography (general)
    "country", "region", "city", "state", "province", "address",
    "location", "area", "zone", "territory",

    # Basic math/statistics
    "count", "total", "sum", "average", "mean", "median", "max", "min",
    "percent", "percentage", "ratio", "rate", "proportion",

    # Boolean/status
    "status", "type", "category", "class", "group", "flag", "active",
    "enabled", "disabled", "valid", "invalid",

    # Generic descriptors
    "description", "comment", "note", "notes", "text", "content",
    "message", "remark", "summary",

    # File/path
    "file", "path", "url", "link", "image", "photo", "document",

    # User-related
    "user", "username", "email", "phone", "password", "role",

    # Generic measures
    "size", "length", "width", "height", "weight", "quantity", "amount",
    "value", "price", "cost", "score", "level",
}

# Tier 1: domain-specific terms worth injecting
TIER_1_PATTERNS: List[str] = [
    # Finance - basic
    "revenue", "profit", "margin", "ebitda", "roi", "roe", "eps",
    "market_cap", "volatility", "dividend", "yield",
    # Finance - OHLC
    "open", "high", "low", "close", "volume", "adj_close",
    "bid", "ask", "spread", "return",
    # ESG
    "esg", "environmental", "social", "governance", "carbon",
    "emission", "sustainability", "renewable",
    # Technology
    "latency", "throughput", "bandwidth", "cpu", "gpu", "memory",
    # Healthcare
    "diagnosis", "treatment", "dosage", "symptom", "patient",
    "icu", "hospital", "mortality", "incidence", "prevalence",
    # Research
    "hypothesis", "experiment", "variable", "control", "sample",
]

# Tier 2: entity-linking indicators
TIER_2_INDICATORS: List[str] = [
    "ticker", "symbol", "stock", "company", "organization", "brand",
    "product", "sku", "isin", "cusip", "lei",
]


class Tier0Filter:
    """Classifies concepts by knowledge tier and decides injection priority."""

    def __init__(
        self,
        blacklist: Optional[Set[str]] = None,
        tier1_patterns: Optional[List[str]] = None,
        tier2_indicators: Optional[List[str]] = None
    ):
        self.blacklist = blacklist or TIER_0_BLACKLIST
        self.tier1_patterns = tier1_patterns or TIER_1_PATTERNS
        self.tier2_indicators = tier2_indicators or TIER_2_INDICATORS

    def classify_concept(self, concept_name: str, db_conn: Optional[Any] = None) -> ConceptTier:
        """Determine the knowledge tier of a concept (checks DB labels first)."""
        name_lower = concept_name.lower().strip()

        # Check database labels if available
        if db_conn:
            try:
                cursor = db_conn.cursor()
                cursor.execute("SELECT tier FROM concepts WHERE lower(display_name) = ?", (name_lower,))
                row = cursor.fetchone()
                if row:
                    db_tier = row[0]
                    if db_tier == 0: return ConceptTier.TIER_0
                    if db_tier == 1: return ConceptTier.TIER_1
                    if db_tier == 2: return ConceptTier.TIER_2
            except:
                pass

        # Fallback: rule-based classification
        if name_lower in self.blacklist:
            return ConceptTier.TIER_0

        normalized = self._normalize_name(name_lower)
        if normalized in self.blacklist:
            return ConceptTier.TIER_0

        for indicator in self.tier2_indicators:
            if indicator in name_lower:
                return ConceptTier.TIER_2

        for pattern in self.tier1_patterns:
            if pattern in name_lower:
                return ConceptTier.TIER_1

        return ConceptTier.UNKNOWN

    def filter(self, concept_name: str, kg_score: float = 0.0) -> FilterResult:
        """Decide whether a concept should be injected into the prompt."""
        tier = self.classify_concept(concept_name)

        if tier == ConceptTier.TIER_0:
            return FilterResult(
                concept_name=concept_name,
                tier=tier,
                should_inject=False,
                weight=0.0,
                reason="Common sense concept (LLM already knows)"
            )

        elif tier == ConceptTier.TIER_1:
            return FilterResult(
                concept_name=concept_name,
                tier=tier,
                should_inject=True,
                weight=0.8,
                reason="Domain-specific term (valuable)"
            )

        elif tier == ConceptTier.TIER_2:
            return FilterResult(
                concept_name=concept_name,
                tier=tier,
                should_inject=True,
                weight=1.0,
                reason="Entity linking candidate (high value)"
            )

        else:
            # Unknown tier: decide based on KG score
            weight = min(1.0, kg_score) if kg_score > 0.7 else 0.5
            return FilterResult(
                concept_name=concept_name,
                tier=tier,
                should_inject=kg_score > 0.5,
                weight=weight,
                reason=f"Unknown concept (score-based: {kg_score:.2f})"
            )

    def filter_batch(
        self,
        concept_names: List[str],
        kg_scores: Optional[List[float]] = None
    ) -> List[FilterResult]:
        """Filter multiple concepts at once."""
        results = []
        for i, name in enumerate(concept_names):
            score = kg_scores[i] if kg_scores and i < len(kg_scores) else 0.0
            results.append(self.filter(name, score))
        return results

    def filter_kg_matches(
        self,
        kg_matches: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Annotate KG match results with tier and injection decisions."""
        filtered = []
        for match in kg_matches:
            if match.get("status") != "matched":
                filtered.append(match)
                continue

            concept = match.get("concept", {})
            concept_name = concept.get("display_name", "")
            kg_score = match.get("score", 0.0)

            result = self.filter(concept_name, kg_score)

            enhanced_match = match.copy()
            enhanced_match["tier"] = result.tier.value
            enhanced_match["should_inject"] = result.should_inject
            enhanced_match["injection_weight"] = result.weight
            enhanced_match["filter_reason"] = result.reason

            filtered.append(enhanced_match)

        return filtered

    def _normalize_name(self, name: str) -> str:
        """Strip common suffixes and underscores for matching against the blacklist."""
        suffixes = ['_id', '_name', '_code', '_type', '_date', '_time', 's']
        for suffix in suffixes:
            if name.endswith(suffix) and len(name) > len(suffix):
                name = name[:-len(suffix)]

        name = name.replace('_', '')
        return name

    def print_summary(self, results: List[FilterResult]) -> None:
        """Print a filtering summary."""
        tier_counts = {}
        inject_count = 0

        for r in results:
            tier_counts[r.tier.value] = tier_counts.get(r.tier.value, 0) + 1
            if r.should_inject:
                inject_count += 1

        print(f"\nTier 0 Filter Summary:")
        print(f"   Total concepts: {len(results)}")
        print(f"   Will inject: {inject_count} ({inject_count/len(results)*100:.1f}%)")
        print(f"   Will skip: {len(results) - inject_count} ({(len(results)-inject_count)/len(results)*100:.1f}%)")

        print(f"\n   By Tier:")
        for tier, count in sorted(tier_counts.items()):
            print(f"     - {tier}: {count}")


def is_tier0_concept(concept_name: str) -> bool:
    """Check if a concept is Tier 0 (common sense)."""
    filter = Tier0Filter()
    return filter.classify_concept(concept_name) == ConceptTier.TIER_0


def should_inject_concept(concept_name: str, kg_score: float = 0.0) -> bool:
    """Check if a concept should be injected into the prompt."""
    filter = Tier0Filter()
    return filter.filter(concept_name, kg_score).should_inject


if __name__ == "__main__":
    print("Testing Tier 0 Filter\n")

    filter = Tier0Filter()

    test_concepts = [
        ("Year", 0.95),
        ("Name", 0.90),
        ("Country", 0.92),
        ("Date", 0.88),
        ("Status", 0.85),
        ("Revenue", 0.90),
        ("Profit Margin", 0.96),
        ("ESG Score", 0.88),
        ("Carbon Emissions", 0.85),
        ("Ticker Symbol", 0.95),
        ("Company Name", 0.90),
        ("Stock Price", 0.88),
        ("GrowthRate", 0.75),
        ("MarketCap", 0.82),
        ("RandomColumn", 0.30),
    ]

    print("=" * 75)
    print(f"{'Concept':<20} {'Score':<8} {'Tier':<12} {'Inject':<8} {'Weight':<8}")
    print("=" * 75)

    results = []
    for concept, score in test_concepts:
        result = filter.filter(concept, score)
        results.append(result)
        inject_str = "YES" if result.should_inject else "NO"
        print(f"{concept:<20} {score:<8.2f} {result.tier.value:<12} {inject_str:<8} {result.weight:<8.2f}")

    filter.print_summary(results)
