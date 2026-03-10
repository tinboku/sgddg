#!/usr/bin/env python3
"""Cost Tracker - tracks and analyzes API costs in real-time."""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class APICall:
    """Record of a single API call."""
    timestamp: float
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cached: bool = False
    component: str = "unknown"


class CostTracker:
    """Tracks API costs with per-component and per-model analytics."""

    # Pricing (as of 2026-02, per 1M tokens)
    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "claude-opus-4": {"input": 15.00, "output": 75.00},
        "claude-sonnet-4": {"input": 3.00, "output": 15.00},
        "claude-haiku-4": {"input": 0.25, "output": 1.25},
    }

    def __init__(self):
        self.calls: List[APICall] = []
        self.start_time = time.time()

    def add_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached: bool = False,
        component: str = "unknown"
    ) -> float:
        """
        Record an API call and return its cost.

        Args:
            model: Model name (e.g., "gpt-4o-mini")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cached: Whether this was a cached call
            component: Component that made the call

        Returns:
            Cost in USD
        """
        if cached:
            cost = 0.0  # Cached calls are free
        else:
            pricing = self.PRICING.get(model, {"input": 1.0, "output": 3.0})
            cost = (
                (input_tokens / 1_000_000) * pricing["input"] +
                (output_tokens / 1_000_000) * pricing["output"]
            )

        call = APICall(
            timestamp=time.time(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            cached=cached,
            component=component
        )

        self.calls.append(call)
        return cost

    def get_summary(self) -> Dict:
        """Get cost summary."""
        if not self.calls:
            return {
                "total_calls": 0,
                "total_cost": 0.0,
                "total_tokens": 0,
                "cached_calls": 0,
                "cache_savings": 0.0
            }

        total_cost = sum(c.cost_usd for c in self.calls)
        total_tokens = sum(c.input_tokens + c.output_tokens for c in self.calls)
        cached_calls = sum(1 for c in self.calls if c.cached)

        # Estimate cache savings (assume cached calls would have cost average)
        non_cached_calls = [c for c in self.calls if not c.cached]
        if non_cached_calls:
            avg_cost_per_call = sum(c.cost_usd for c in non_cached_calls) / len(non_cached_calls)
            cache_savings = cached_calls * avg_cost_per_call
        else:
            cache_savings = 0.0

        # Group by component
        by_component = {}
        for call in self.calls:
            if call.component not in by_component:
                by_component[call.component] = {
                    "calls": 0,
                    "cost": 0.0,
                    "tokens": 0
                }
            by_component[call.component]["calls"] += 1
            by_component[call.component]["cost"] += call.cost_usd
            by_component[call.component]["tokens"] += call.input_tokens + call.output_tokens

        # Group by model
        by_model = {}
        for call in self.calls:
            if call.model not in by_model:
                by_model[call.model] = {
                    "calls": 0,
                    "cost": 0.0,
                    "tokens": 0
                }
            by_model[call.model]["calls"] += 1
            by_model[call.model]["cost"] += call.cost_usd
            by_model[call.model]["tokens"] += call.input_tokens + call.output_tokens

        elapsed_time = time.time() - self.start_time

        return {
            "total_calls": len(self.calls),
            "total_cost": round(total_cost, 4),
            "total_tokens": total_tokens,
            "cached_calls": cached_calls,
            "cache_savings": round(cache_savings, 4),
            "cache_hit_rate": round(cached_calls / len(self.calls), 3) if self.calls else 0.0,
            "by_component": by_component,
            "by_model": by_model,
            "elapsed_seconds": round(elapsed_time, 2)
        }

    def print_summary(self):
        """Print a formatted cost summary."""
        summary = self.get_summary()

        print("\n" + "="*70)
        print("API COST SUMMARY")
        print("="*70)

        print(f"\nOverall Statistics:")
        print(f"  Total API Calls: {summary['total_calls']}")
        print(f"  Total Cost: ${summary['total_cost']:.4f}")
        print(f"  Total Tokens: {summary['total_tokens']:,}")
        print(f"  Elapsed Time: {summary['elapsed_seconds']:.1f}s")

        print(f"\nCache Performance:")
        print(f"  Cached Calls: {summary['cached_calls']}")
        print(f"  Cache Hit Rate: {summary['cache_hit_rate']:.1%}")
        print(f"  Estimated Savings: ${summary['cache_savings']:.4f}")
        print(f"  Effective Cost: ${summary['total_cost'] - summary['cache_savings']:.4f}")

        print(f"\nCost by Component:")
        for component, stats in sorted(
            summary['by_component'].items(),
            key=lambda x: x[1]['cost'],
            reverse=True
        ):
            print(f"  {component}:")
            print(f"    Calls: {stats['calls']}, "
                  f"Cost: ${stats['cost']:.4f}, "
                  f"Tokens: {stats['tokens']:,}")

        print(f"\nCost by Model:")
        for model, stats in sorted(
            summary['by_model'].items(),
            key=lambda x: x[1]['cost'],
            reverse=True
        ):
            print(f"  {model}:")
            print(f"    Calls: {stats['calls']}, "
                  f"Cost: ${stats['cost']:.4f}, "
                  f"Tokens: {stats['tokens']:,}")

        print("\n" + "="*70 + "\n")

    def export_to_json(self, filepath: str):
        """Export detailed cost data to JSON."""
        data = {
            "summary": self.get_summary(),
            "calls": [
                {
                    "timestamp": datetime.fromtimestamp(c.timestamp).isoformat(),
                    "model": c.model,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "cost_usd": c.cost_usd,
                    "cached": c.cached,
                    "component": c.component
                }
                for c in self.calls
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Cost report exported to: {filepath}")

    def get_recommendations(self) -> List[str]:
        """Get cost optimization recommendations."""
        recommendations = []
        summary = self.get_summary()

        # Check cache performance
        if summary['cache_hit_rate'] < 0.3 and summary['total_calls'] > 10:
            recommendations.append(
                "Low cache hit rate (<30%). Consider enabling caching or increasing TTL."
            )

        # Check model usage
        expensive_models = ["gpt-4o", "gpt-4-turbo", "claude-opus-4"]
        for model, stats in summary['by_model'].items():
            if model in expensive_models and stats['calls'] > 5:
                recommendations.append(
                    f"Consider using a cheaper model for {model} calls "
                    f"(${stats['cost']:.4f} spent)"
                )

        # Check token usage
        avg_tokens = summary['total_tokens'] / summary['total_calls'] if summary['total_calls'] > 0 else 0
        if avg_tokens > 3000:
            recommendations.append(
                f"High average tokens per call ({avg_tokens:.0f}). "
                "Consider prompt optimization or batch processing."
            )

        # Component-specific recommendations
        by_component = summary['by_component']
        if 'semantic_profiler' in by_component:
            sem_stats = by_component['semantic_profiler']
            if sem_stats['cost'] > summary['total_cost'] * 0.5:
                recommendations.append(
                    "SemanticProfiler is >50% of cost. "
                    "Use batch processing (profile_dataset_batch) instead of per-column calls."
                )

        if not recommendations:
            recommendations.append("Cost optimization looks good!")

        return recommendations


# Global tracker instance
_global_tracker = None

def get_global_tracker() -> CostTracker:
    """Get or create global cost tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = CostTracker()
    return _global_tracker


# Demo/Test
if __name__ == "__main__":
    print("Cost Tracker Demo\n")

    tracker = CostTracker()

    # Simulate API calls
    print("Simulating API calls...\n")

    # SemanticProfiler calls
    for i in range(10):
        tracker.add_call(
            model="gpt-4o-mini",
            input_tokens=800,
            output_tokens=200,
            component="semantic_profiler"
        )

    # Some cached calls
    for i in range(5):
        tracker.add_call(
            model="gpt-4o-mini",
            input_tokens=800,
            output_tokens=200,
            cached=True,
            component="semantic_profiler"
        )

    # UFD generation
    tracker.add_call(
        model="gpt-4o-mini",
        input_tokens=2000,
        output_tokens=500,
        component="ufd_generator"
    )

    # SFD generation
    tracker.add_call(
        model="gpt-4o-mini",
        input_tokens=1500,
        output_tokens=400,
        component="sfd_generator"
    )

    # Reranking
    for i in range(3):
        tracker.add_call(
            model="gpt-4o-mini",
            input_tokens=600,
            output_tokens=100,
            component="reranker"
        )

    # Print summary
    tracker.print_summary()

    # Get recommendations
    print("Optimization Recommendations:")
    for rec in tracker.get_recommendations():
        print(f"  {rec}")

    print("\nDemo complete!")
