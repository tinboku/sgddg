"""
Conflict resolver and semantic strength labeler for SGDDG.

Detects statistical constraint conflicts between KG matches and column profiles,
and labels matches with semantic strength (EXACT, RELATED, CONTEXTUAL, UNCERTAIN).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Set
import re


class ConflictType(Enum):
    """Types of conflicts between KG concepts and statistical profiles."""
    NONE = "none"
    TYPE_MISMATCH = "type_mismatch"
    RANGE_VIOLATION = "range_violation"
    DISTRIBUTION_CONFLICT = "distribution_conflict"
    SEMANTIC_CONFLICT = "semantic_conflict"
    DOMAIN_MISMATCH = "domain_mismatch"


class Resolution(Enum):
    """Conflict resolution strategy."""
    ACCEPT = "accept"
    DOWNWEIGHT = "downweight"
    REJECT = "reject"


@dataclass
class ConflictResult:
    """Result of conflict detection."""
    has_conflict: bool
    conflict_type: ConflictType
    resolution: Resolution
    confidence_penalty: float  # 0.0 - 1.0
    reason: str
    evidence: Dict[str, Any]


# Domain keyword constraints for cross-domain conflict detection
DOMAIN_CONSTRAINTS = {
    "temporal": ["date", "time", "year", "month", "day", "hour", "timestamp", "period", "duration"],
    "geographic": ["country", "city", "region", "province", "address", "location", "lat", "lon", "line"],
    "financial": ["price", "amount", "revenue", "profit", "cost", "market", "stock", "balance", "open", "high", "low", "close", "vol", "ticker", "dividend"],
    "biological": ["species", "genus", "gene", "protein", "organism", "cell"],
    "demographic": ["age", "gender", "race", "ethnicity", "education", "income"],
    "meteorology": ["weather", "climate", "temperature", "pressure", "humidity", "wind", "precipitation", "storm", "hurricane", "system", "forecast"],
}

# Expected data types for known concepts
CONCEPT_TYPE_HINTS = {
    "price": {"expected_types": ["float64", "int64", "float"], "numeric": True},
    "amount": {"expected_types": ["float64", "int64", "float"], "numeric": True},
    "rate": {"expected_types": ["float64", "float"], "numeric": True},
    "percentage": {"expected_types": ["float64", "float"], "numeric": True, "range": (0, 100)},
    "count": {"expected_types": ["int64", "int"], "numeric": True, "non_negative": True},
    "year": {"expected_types": ["int64", "int"], "numeric": True, "range": (1900, 2100)},
    "age": {"expected_types": ["int64", "float64"], "numeric": True, "range": (0, 150)},

    "name": {"expected_types": ["object", "str", "string"], "numeric": False},
    "description": {"expected_types": ["object", "str", "string"], "numeric": False},
    "code": {"expected_types": ["object", "str", "string"], "numeric": False},
    "category": {"expected_types": ["object", "str", "string"], "numeric": False},

    "date": {"expected_types": ["datetime64", "object"], "datetime": True},
    "timestamp": {"expected_types": ["datetime64", "int64"], "datetime": True},

    "flag": {"expected_types": ["bool", "int64"], "boolean": True},
    "active": {"expected_types": ["bool", "int64"], "boolean": True},
}


class ConflictResolver:
    """Detects conflicts between KG matches and statistical column profiles."""

    def __init__(
        self,
        type_hints: Optional[Dict] = None,
        strict_mode: bool = False
    ):
        self.type_hints = type_hints or CONCEPT_TYPE_HINTS
        self.strict_mode = strict_mode

    def check_conflict(
        self,
        kg_concept: str,
        kg_score: float,
        stats_profile: Dict[str, Any]
    ) -> ConflictResult:
        """Check for conflicts between a KG concept and statistical profile."""
        concept_lower = kg_concept.lower()
        conflicts = []
        evidence = {}

        type_hint = self._get_type_hint(concept_lower)

        if type_hint:
            type_conflict = self._check_type_match(stats_profile, type_hint)
            if type_conflict:
                conflicts.append(type_conflict)
                evidence["type_conflict"] = type_conflict

            range_conflict = self._check_range_constraint(stats_profile, type_hint)
            if range_conflict:
                conflicts.append(range_conflict)
                evidence["range_conflict"] = range_conflict

            dist_conflict = self._check_distribution(stats_profile, type_hint)
            if dist_conflict:
                conflicts.append(dist_conflict)
                evidence["distribution_conflict"] = dist_conflict

        domain_conflict = self._check_domain_conflict(concept_lower, stats_profile)
        if domain_conflict:
            conflicts.append(domain_conflict)
            evidence["domain_conflict"] = domain_conflict

        semantic_conflict = self._check_semantic_conflict(
            concept_lower, stats_profile
        )
        if semantic_conflict:
            conflicts.append(semantic_conflict)
            evidence["semantic_conflict"] = semantic_conflict

        if not conflicts:
            return ConflictResult(
                has_conflict=False,
                conflict_type=ConflictType.NONE,
                resolution=Resolution.ACCEPT,
                confidence_penalty=0.0,
                reason="No conflicts detected",
                evidence=evidence
            )

        total_penalty = sum(c.get("penalty", 0.1) for c in conflicts)
        total_penalty = min(1.0, total_penalty)

        primary_conflict = max(conflicts, key=lambda x: x.get("penalty", 0))
        conflict_type = ConflictType(primary_conflict.get("type", "semantic_conflict"))

        if self.strict_mode or total_penalty > 0.5:
            resolution = Resolution.REJECT
        elif total_penalty > 0.2:
            resolution = Resolution.DOWNWEIGHT
        else:
            resolution = Resolution.ACCEPT

        return ConflictResult(
            has_conflict=True,
            conflict_type=conflict_type,
            resolution=resolution,
            confidence_penalty=total_penalty,
            reason="; ".join(c.get("reason", "") for c in conflicts),
            evidence=evidence
        )

    def _get_type_hint(self, concept: str) -> Optional[Dict]:
        """Look up type hints for a concept (exact then partial match)."""
        if concept in self.type_hints:
            return self.type_hints[concept]

        for keyword, hint in self.type_hints.items():
            if keyword in concept:
                return hint

        return None

    def _check_type_match(
        self,
        stats_profile: Dict[str, Any],
        type_hint: Dict
    ) -> Optional[Dict]:
        """Check whether the actual data type matches expected types."""
        actual_type = str(stats_profile.get("data_type", "unknown")).lower()
        expected_types = type_hint.get("expected_types", [])

        expected_lower = [t.lower() for t in expected_types]
        type_matched = any(exp in actual_type for exp in expected_lower)

        if not type_matched and expected_types:
            return {
                "type": "type_mismatch",
                "reason": f"Expected {expected_types}, got {actual_type}",
                "penalty": 0.3,
            }

        if type_hint.get("numeric") and "object" in actual_type:
            return {
                "type": "type_mismatch",
                "reason": f"Concept implies numeric, but data is {actual_type}",
                "penalty": 0.25,
            }

        return None

    def _check_range_constraint(
        self,
        stats_profile: Dict[str, Any],
        type_hint: Dict
    ) -> Optional[Dict]:
        """Check whether values fall within expected range."""
        expected_range = type_hint.get("range")
        if not expected_range:
            return None

        actual_min = stats_profile.get("min")
        actual_max = stats_profile.get("max")
        if actual_min is None or actual_max is None:
            return None

        exp_min, exp_max = expected_range

        if actual_min < exp_min * 0.5 or actual_max > exp_max * 2:
            return {
                "type": "range_violation",
                "reason": f"Value range [{actual_min}, {actual_max}] violates expected [{exp_min}, {exp_max}]",
                "penalty": 0.3,
            }

        if type_hint.get("non_negative") and actual_min < 0:
            return {
                "type": "range_violation",
                "reason": f"Concept implies non-negative, but min={actual_min}",
                "penalty": 0.2,
            }

        return None

    def _check_distribution(
        self,
        stats_profile: Dict[str, Any],
        type_hint: Dict
    ) -> Optional[Dict]:
        """Check distribution characteristics (cardinality, skewness)."""
        # Boolean concept but too many unique values
        if type_hint.get("boolean"):
            unique_count = stats_profile.get("unique_count", 0)
            if unique_count > 3:
                return {
                    "type": "distribution_conflict",
                    "reason": f"Concept implies boolean, but has {unique_count} unique values",
                    "penalty": 0.25,
                }

        # ID concept but low uniqueness
        if "id" in type_hint.get("expected_types", []) or "code" in type_hint.get("expected_types", []):
            unique_ratio = stats_profile.get("unique_ratio", 1.0)
            if unique_ratio < 0.5:
                return {
                    "type": "distribution_conflict",
                    "reason": f"Concept implies identifier, but unique ratio is low ({unique_ratio:.2f})",
                    "penalty": 0.2,
                }

        # Skewness check for financial values
        skewness = stats_profile.get("skewness")
        if skewness is not None and type_hint.get("numeric"):
            if "price" in type_hint.get("expected_types", []) and skewness < -1:
                return {
                    "type": "distribution_conflict",
                    "reason": f"Financial values (price) are usually right-skewed, but data is left-skewed ({skewness:.2f})",
                    "penalty": 0.15,
                }

        return None

    def _check_domain_conflict(
        self,
        concept: str,
        stats_profile: Dict[str, Any]
    ) -> Optional[Dict]:
        """Check for cross-domain conflicts (e.g., temporal column matched to geographic concept)."""
        column_name = str(stats_profile.get("column_name", "")).lower()

        col_domains = self._get_domains(column_name)
        concept_domains = self._get_domains(concept)

        if not col_domains or not concept_domains:
            return None

        # Special case: temporal column matched to geographic concept
        if "temporal" in col_domains and "geographic" in concept_domains:
            strong_geo = ["line", "border", "coast", "ocean", "sea"]
            if any(w in concept.lower() for w in strong_geo) and not any(w in column_name for w in strong_geo):
                return {
                    "type": "domain_mismatch",
                    "reason": f"Semantic drift: Concept '{concept}' has geographic markers not present in column '{column_name}'",
                    "penalty": 0.4,
                }

        intersection = col_domains.intersection(concept_domains)
        if not intersection:
            return {
                "type": "domain_mismatch",
                "reason": f"Domain mismatch: Column '{column_name}' is {list(col_domains)}, but concept '{concept}' is {list(concept_domains)}",
                "penalty": 0.5,
            }

        return None

    def _get_domains(self, text: str) -> Set[str]:
        """Identify which domains a text belongs to using word-boundary matching."""
        found = set()
        for domain, keywords in DOMAIN_CONSTRAINTS.items():
            for kw in keywords:
                if re.search(rf"\b{kw}\b", text, re.IGNORECASE):
                    found.add(domain)
                    break
        return found

    def _check_semantic_conflict(
        self,
        concept: str,
        stats_profile: Dict[str, Any]
    ) -> Optional[Dict]:
        """Check for known mismatching patterns (e.g., 'Years of Schooling' vs calendar year)."""
        mismatches = [
            ("years of schooling", lambda p: p.get("min", 0) > 1900 and p.get("max", 0) < 2100,
             "Matched 'Years of Schooling' but data looks like calendar years"),
            ("age", lambda p: p.get("max", 0) > 150 or p.get("min", 0) < 0,
             "Matched 'Age' but value range is not humanly possible"),
        ]

        for pattern, check_func, reason in mismatches:
            if pattern in concept and check_func(stats_profile):
                return {
                    "type": "semantic_conflict",
                    "reason": reason,
                    "penalty": 0.4,
                }

        return None

    def filter_matches(
        self,
        kg_matches: List[Dict[str, Any]],
        stats_profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Filter and annotate KG matches with conflict information."""
        filtered = []

        for match in kg_matches:
            concept = match.get("concept", {}).get("display_name", "")
            score = match.get("score", 0.0)

            result = self.check_conflict(concept, score, stats_profile)

            enhanced = match.copy()
            enhanced["conflict_result"] = {
                "has_conflict": result.has_conflict,
                "conflict_type": result.conflict_type.value,
                "resolution": result.resolution.value,
                "penalty": result.confidence_penalty,
                "reason": result.reason,
            }

            if result.resolution == Resolution.REJECT:
                enhanced["filtered"] = True
                enhanced["original_score"] = score
                enhanced["adjusted_score"] = 0.0
            elif result.resolution == Resolution.DOWNWEIGHT:
                enhanced["filtered"] = False
                enhanced["original_score"] = score
                enhanced["adjusted_score"] = score * (1 - result.confidence_penalty)
            else:
                enhanced["filtered"] = False
                enhanced["original_score"] = score
                enhanced["adjusted_score"] = score

            filtered.append(enhanced)

        filtered.sort(key=lambda x: x.get("adjusted_score", 0), reverse=True)
        return filtered

    def print_conflict_report(
        self,
        kg_matches: List[Dict],
        stats_profile: Dict
    ):
        """Print a conflict detection report."""
        print("\nConflict Detection Report")
        print("=" * 60)

        for match in kg_matches:
            concept = match.get("concept", {}).get("display_name", "Unknown")
            score = match.get("score", 0.0)

            result = self.check_conflict(concept, score, stats_profile)

            if result.has_conflict:
                icon = "[WARN]" if result.resolution == Resolution.DOWNWEIGHT else "[FAIL]"
                print(f"{icon} {concept} (score: {score:.2f})")
                print(f"   Conflict: {result.conflict_type.value}")
                print(f"   Resolution: {result.resolution.value}")
                print(f"   Penalty: {result.confidence_penalty:.2f}")
                print(f"   Reason: {result.reason}")
            else:
                print(f"[OK] {concept} (score: {score:.2f}) - No conflicts")
            print()


# === Semantic Strength Labeler ===

class SemanticStrength(Enum):
    """Semantic match strength."""
    EXACT = "EXACT"
    RELATED = "RELATED"
    CONTEXTUAL = "CONTEXTUAL"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class LabeledMatch:
    """A KG match annotated with semantic strength."""
    concept: str
    original_score: float
    adjusted_score: float
    semantic_strength: SemanticStrength
    label_reason: str
    conflict_info: Optional[ConflictResult] = None


class SemanticStrengthLabeler:
    """Labels KG matches with semantic strength based on score and conflict analysis."""

    def __init__(
        self,
        exact_threshold: float = 0.9,
        related_threshold: float = 0.7,
        contextual_threshold: float = 0.5
    ):
        self.exact_threshold = exact_threshold
        self.related_threshold = related_threshold
        self.contextual_threshold = contextual_threshold
        self.conflict_resolver = ConflictResolver()

    def label(
        self,
        kg_concept: str,
        kg_score: float,
        stats_profile: Dict[str, Any]
    ) -> LabeledMatch:
        """Label a single KG match with semantic strength."""
        conflict = self.conflict_resolver.check_conflict(
            kg_concept, kg_score, stats_profile
        )

        adjusted_score = kg_score * (1 - conflict.confidence_penalty)

        if conflict.resolution == Resolution.REJECT:
            strength = SemanticStrength.UNCERTAIN
            reason = f"Rejected due to conflict: {conflict.reason}"
        elif adjusted_score >= self.exact_threshold:
            strength = SemanticStrength.EXACT
            reason = f"High confidence match (score: {adjusted_score:.2f})"
        elif adjusted_score >= self.related_threshold:
            strength = SemanticStrength.RELATED
            reason = f"Related concept (score: {adjusted_score:.2f})"
        elif adjusted_score >= self.contextual_threshold:
            strength = SemanticStrength.CONTEXTUAL
            reason = f"Contextual inference (score: {adjusted_score:.2f})"
        else:
            strength = SemanticStrength.UNCERTAIN
            reason = f"Low confidence (score: {adjusted_score:.2f})"

        return LabeledMatch(
            concept=kg_concept,
            original_score=kg_score,
            adjusted_score=adjusted_score,
            semantic_strength=strength,
            label_reason=reason,
            conflict_info=conflict
        )

    def label_batch(
        self,
        kg_matches: List[Tuple[str, float]],
        stats_profile: Dict[str, Any]
    ) -> List[LabeledMatch]:
        """Label multiple KG matches at once."""
        return [
            self.label(concept, score, stats_profile)
            for concept, score in kg_matches
        ]

    def format_for_prompt(self, labeled_matches: List[LabeledMatch]) -> str:
        """Format labeled matches for LLM prompt injection."""
        if not labeled_matches:
            return "No KG matches available."

        lines = ["[Knowledge Graph Matches]"]

        for match in labeled_matches:
            if match.semantic_strength == SemanticStrength.UNCERTAIN:
                continue

            lines.append(
                f"  [{match.semantic_strength.value}] {match.concept} "
                f"(confidence: {match.adjusted_score:.2f})"
            )

        return "\n".join(lines)


def check_kg_conflict(
    kg_concept: str,
    stats_profile: Dict[str, Any]
) -> ConflictResult:
    """Convenience: check a KG concept for conflicts."""
    resolver = ConflictResolver()
    return resolver.check_conflict(kg_concept, 0.8, stats_profile)


def label_kg_match(
    kg_concept: str,
    kg_score: float,
    stats_profile: Dict[str, Any]
) -> LabeledMatch:
    """Convenience: label a KG match with semantic strength."""
    labeler = SemanticStrengthLabeler()
    return labeler.label(kg_concept, kg_score, stats_profile)


if __name__ == "__main__":
    print("Testing Conflict Resolver & Semantic Strength Labeler\n")

    resolver = ConflictResolver()
    labeler = SemanticStrengthLabeler()

    test_cases = [
        {
            "concept": "Year",
            "score": 0.95,
            "profile": {
                "data_type": "int64",
                "min": 2018,
                "max": 2024,
                "unique_count": 7
            },
            "expected": "EXACT"
        },
        {
            "concept": "Years of Schooling",
            "score": 0.90,
            "profile": {
                "data_type": "int64",
                "min": 2018,
                "max": 2024,
                "unique_count": 7
            },
            "expected": "UNCERTAIN or CONTEXTUAL"
        },
        {
            "concept": "Price",
            "score": 0.85,
            "profile": {
                "data_type": "object",
                "unique_count": 100
            },
            "expected": "RELATED or CONTEXTUAL"
        },
        {
            "concept": "Percentage",
            "score": 0.88,
            "profile": {
                "data_type": "float64",
                "min": 0.0,
                "max": 100.0
            },
            "expected": "EXACT or RELATED"
        },
    ]

    print("=" * 70)
    print(f"{'Concept':<25} {'Score':<8} {'Strength':<12} {'Adjusted':<10}")
    print("=" * 70)

    for tc in test_cases:
        result = labeler.label(tc["concept"], tc["score"], tc["profile"])

        conflict_icon = "[WARN]" if result.conflict_info and result.conflict_info.has_conflict else "[OK]"

        print(f"{conflict_icon} {tc['concept']:<22} {tc['score']:<8.2f} {result.semantic_strength.value:<12} {result.adjusted_score:<10.2f}")

        if result.conflict_info and result.conflict_info.has_conflict:
            print(f"   └─ Conflict: {result.conflict_info.reason}")

    print("=" * 70)

    print("\nSample Prompt Format:")
    print("-" * 40)

    sample_matches = [
        labeler.label("Revenue", 0.95, {"data_type": "float64", "min": 0}),
        labeler.label("Profit Margin", 0.88, {"data_type": "float64"}),
        labeler.label("Year", 0.70, {"data_type": "int64", "min": 2020, "max": 2024}),
    ]

    print(labeler.format_for_prompt(sample_matches))
