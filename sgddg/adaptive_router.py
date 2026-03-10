"""
Adaptive knowledge router for SGDDG.

Dynamically adjusts KG injection volume based on statistical profile confidence:
high confidence => minimal KG, low confidence => full KG injection.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional


class ConfidenceLevel(Enum):
    """Statistical confidence level."""
    HIGH = "high"        # > threshold
    MEDIUM = "medium"    # between thresholds
    LOW = "low"          # < threshold


class InjectionPolicy(Enum):
    """KG injection policy."""
    SKIP = "skip"
    MINIMAL = "minimal"    # 1 concept
    STANDARD = "standard"  # 2 concepts
    FULL = "full"          # 3 concepts


@dataclass
class RoutingDecision:
    """Routing decision for a single column."""
    confidence_level: ConfidenceLevel
    confidence_score: float
    injection_policy: InjectionPolicy
    max_kg_concepts: int
    reason: str


# Semantic type clarity scores
SEMANTIC_TYPE_SCORES = {
    'datetime': 0.40,
    'date': 0.40,
    'time': 0.35,
    'email': 0.40,
    'url': 0.40,
    'phone': 0.35,
    'currency': 0.35,
    'percentage': 0.30,
    'integer': 0.20,
    'float': 0.15,
    'boolean': 0.35,
    'string': 0.10,
    'object': 0.10,
    'unknown': 0.0,
}

# Data quality bonus scores
QUALITY_BONUSES = {
    'low_null_rate': 0.15,
    'reasonable_unique': 0.10,
    'has_distribution': 0.15,
    'has_samples': 0.10,
}


class AdaptiveRouter:
    """Routes columns to appropriate KG injection levels based on profile confidence."""

    def __init__(
        self,
        high_threshold: float = 0.8,
        low_threshold: float = 0.5,
        enable_skip: bool = True
    ):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.enable_skip = enable_skip

    def calculate_confidence(self, stats_profile: Dict[str, Any]) -> float:
        """Compute a confidence score from the statistical profile.

        Considers semantic type clarity, null rate, unique ratio,
        distribution features, and sample availability.
        """
        if not stats_profile:
            return 0.0

        score = 0.0

        # Semantic type clarity
        semantic_type = stats_profile.get('semantic_type', stats_profile.get('data_type', 'unknown'))
        if isinstance(semantic_type, str):
            semantic_type = semantic_type.lower()
        score += SEMANTIC_TYPE_SCORES.get(semantic_type, 0.05)

        # Null rate bonus
        null_rate = stats_profile.get('null_rate', 1.0)
        if null_rate < 0.05:
            score += QUALITY_BONUSES['low_null_rate']
        elif null_rate < 0.20:
            score += QUALITY_BONUSES['low_null_rate'] * 0.5

        # Unique ratio bonus
        unique_ratio = stats_profile.get('unique_ratio', 0)
        if unique_ratio == 0:
            unique_count = stats_profile.get('unique_count', 0)
            total_count = stats_profile.get('total_count', 1)
            if total_count > 0:
                unique_ratio = unique_count / total_count

        if 0.01 < unique_ratio < 0.90:
            score += QUALITY_BONUSES['reasonable_unique']

        # Distribution features bonus
        has_distribution = any([
            stats_profile.get('mean') is not None,
            stats_profile.get('median') is not None,
            stats_profile.get('std') is not None,
            stats_profile.get('min') is not None,
            stats_profile.get('max') is not None,
        ])
        if has_distribution:
            score += QUALITY_BONUSES['has_distribution']

        # Sample values bonus
        samples = stats_profile.get('sample_values', [])
        if samples and len(samples) >= 3:
            score += QUALITY_BONUSES['has_samples']

        # Bonus for explicit value range
        if stats_profile.get('min') is not None and stats_profile.get('max') is not None:
            score += 0.05

        return min(1.0, max(0.0, score))

    def route(self, stats_profile: Dict[str, Any]) -> RoutingDecision:
        """Make a routing decision based on profile confidence."""
        confidence = self.calculate_confidence(stats_profile)

        if confidence > self.high_threshold:
            if self.enable_skip:
                return RoutingDecision(
                    confidence_level=ConfidenceLevel.HIGH,
                    confidence_score=confidence,
                    injection_policy=InjectionPolicy.MINIMAL,
                    max_kg_concepts=1,
                    reason=f"High confidence ({confidence:.2f}): Stats sufficient, minimal KG"
                )
            else:
                return RoutingDecision(
                    confidence_level=ConfidenceLevel.HIGH,
                    confidence_score=confidence,
                    injection_policy=InjectionPolicy.MINIMAL,
                    max_kg_concepts=1,
                    reason=f"High confidence ({confidence:.2f}): Inject top-1 only"
                )

        elif confidence < self.low_threshold:
            return RoutingDecision(
                confidence_level=ConfidenceLevel.LOW,
                confidence_score=confidence,
                injection_policy=InjectionPolicy.FULL,
                max_kg_concepts=3,
                reason=f"Low confidence ({confidence:.2f}): KG as primary source"
            )

        else:
            return RoutingDecision(
                confidence_level=ConfidenceLevel.MEDIUM,
                confidence_score=confidence,
                injection_policy=InjectionPolicy.STANDARD,
                max_kg_concepts=2,
                reason=f"Medium confidence ({confidence:.2f}): Balanced fusion"
            )

    def filter_kg_concepts(
        self,
        kg_matches: List[Dict[str, Any]],
        decision: RoutingDecision
    ) -> List[Dict[str, Any]]:
        """Keep only top-N KG concepts according to the routing decision."""
        if decision.injection_policy == InjectionPolicy.SKIP:
            return []
        return kg_matches[:decision.max_kg_concepts]

    def route_batch(
        self,
        stats_profiles: List[Dict[str, Any]]
    ) -> List[RoutingDecision]:
        """Route multiple columns at once."""
        return [self.route(profile) for profile in stats_profiles]

    def print_summary(self, decisions: List[RoutingDecision]) -> None:
        """Print a routing summary."""
        high_count = sum(1 for d in decisions if d.confidence_level == ConfidenceLevel.HIGH)
        medium_count = sum(1 for d in decisions if d.confidence_level == ConfidenceLevel.MEDIUM)
        low_count = sum(1 for d in decisions if d.confidence_level == ConfidenceLevel.LOW)

        avg_confidence = sum(d.confidence_score for d in decisions) / len(decisions) if decisions else 0

        print(f"\nRouting Summary:")
        print(f"   Total columns: {len(decisions)}")
        print(f"   Average confidence: {avg_confidence:.2f}")
        print(f"   High confidence (minimal KG): {high_count} ({high_count/len(decisions)*100:.1f}%)")
        print(f"   Medium confidence (standard): {medium_count} ({medium_count/len(decisions)*100:.1f}%)")
        print(f"   Low confidence (full KG): {low_count} ({low_count/len(decisions)*100:.1f}%)")


def calculate_confidence(stats_profile: Dict[str, Any]) -> float:
    """Convenience: calculate profile confidence."""
    router = AdaptiveRouter()
    return router.calculate_confidence(stats_profile)


def get_routing_decision(stats_profile: Dict[str, Any]) -> RoutingDecision:
    """Convenience: get a routing decision for a profile."""
    router = AdaptiveRouter()
    return router.route(stats_profile)


if __name__ == "__main__":
    print("Testing Adaptive Router\n")

    router = AdaptiveRouter()

    test_profiles = [
        {
            "column_name": "created_at",
            "semantic_type": "datetime",
            "null_rate": 0.0,
            "unique_ratio": 0.95,
            "sample_values": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "min": "2020-01-01",
            "max": "2024-12-31"
        },
        {
            "column_name": "quantity",
            "semantic_type": "integer",
            "null_rate": 0.02,
            "unique_ratio": 0.30,
            "sample_values": [10, 20, 30],
            "mean": 25.5,
            "std": 10.2
        },
        {
            "column_name": "notes",
            "semantic_type": "string",
            "null_rate": 0.45,
            "unique_ratio": 0.80
        },
        {
            "column_name": "col_x",
            "data_type": "unknown"
        },
    ]

    print("=" * 70)
    print(f"{'Column':<15} {'Confidence':<12} {'Level':<10} {'Policy':<12} {'Max KG':<8}")
    print("=" * 70)

    decisions = []
    for profile in test_profiles:
        decision = router.route(profile)
        decisions.append(decision)
        print(f"{profile['column_name']:<15} {decision.confidence_score:<12.2f} {decision.confidence_level.value:<10} {decision.injection_policy.value:<12} {decision.max_kg_concepts:<8}")

    router.print_summary(decisions)
