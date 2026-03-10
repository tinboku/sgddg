#!/usr/bin/env python3
"""API Cache - caches LLM API call results to reduce costs."""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Callable
from functools import wraps


class APICache:
    """File-based cache for API calls with TTL and size limits."""

    def __init__(
        self,
        cache_dir: str = ".api_cache",
        ttl: int = 86400 * 7,  # 7 days default
        max_size_mb: int = 100
    ):
        """
        Initialize API cache.

        Args:
            cache_dir: Directory to store cache files
            ttl: Time-to-live in seconds (default: 7 days)
            max_size_mb: Maximum cache size in MB
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = ttl
        self.max_size_mb = max_size_mb
        self.stats = {
            "hits": 0,
            "misses": 0,
            "saved_calls": 0,
            "estimated_savings_usd": 0.0
        }

    def _compute_hash(self, *args, **kwargs) -> str:
        """Compute a hash from function arguments."""
        # Create a stable string representation
        key_data = {
            "args": str(args),
            "kwargs": json.dumps(kwargs, sort_keys=True, default=str)
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the cache file path for a given key."""
        return self.cache_dir / f"{cache_key}.json"

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get a cached result.

        Returns:
            Cached data if found and not expired, None otherwise
        """
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            self.stats["misses"] += 1
            return None

        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)

            # Check TTL
            if time.time() - cache_data["timestamp"] > self.ttl:
                # Expired
                cache_path.unlink()
                self.stats["misses"] += 1
                return None

            self.stats["hits"] += 1
            self.stats["saved_calls"] += 1

            # Estimate savings (rough approximation)
            # Assume ~1000 tokens per call, $0.15 per 1M input tokens for gpt-4o-mini
            self.stats["estimated_savings_usd"] += 0.00015

            return cache_data["result"]

        except Exception as e:
            print(f"   Cache read error: {e}")
            self.stats["misses"] += 1
            return None

    def set(self, cache_key: str, result: Any) -> None:
        """Store a result in cache."""
        cache_path = self._get_cache_path(cache_key)

        cache_data = {
            "timestamp": time.time(),
            "result": result
        }

        try:
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)

            # Check cache size
            self._cleanup_if_needed()

        except Exception as e:
            print(f"   Cache write error: {e}")

    def _cleanup_if_needed(self) -> None:
        """Clean up old cache entries if size limit exceeded."""
        total_size = sum(
            f.stat().st_size for f in self.cache_dir.glob("*.json")
        ) / (1024 * 1024)  # Convert to MB

        if total_size > self.max_size_mb:
            # Remove oldest files
            files = sorted(
                self.cache_dir.glob("*.json"),
                key=lambda f: f.stat().st_mtime
            )

            # Remove oldest 20%
            remove_count = max(1, len(files) // 5)
            for f in files[:remove_count]:
                f.unlink()

    def clear(self) -> None:
        """Clear all cache entries."""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "saved_calls": 0,
            "estimated_savings_usd": 0.0
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = (
            self.stats["hits"] / (self.stats["hits"] + self.stats["misses"])
            if (self.stats["hits"] + self.stats["misses"]) > 0
            else 0.0
        )

        return {
            **self.stats,
            "hit_rate": hit_rate,
            "total_requests": self.stats["hits"] + self.stats["misses"]
        }

    def cached_call(
        self,
        func: Callable,
        *args,
        cost_per_call: float = 0.0015,  # Rough estimate for gpt-4o-mini
        **kwargs
    ) -> Any:
        """
        Call a function with caching.

        Args:
            func: Function to call
            cost_per_call: Estimated cost in USD per call
            *args, **kwargs: Arguments to pass to func

        Returns:
            Function result (from cache or fresh call)
        """
        # Compute cache key
        cache_key = self._compute_hash(func.__name__, *args, **kwargs)

        # Try to get from cache
        cached_result = self.get(cache_key)
        if cached_result is not None:
            return cached_result

        # Call function
        result = func(*args, **kwargs)

        # Store in cache
        self.set(cache_key, result)

        return result


def cache_api_call(
    cache: APICache,
    cost_estimate: float = 0.0015
):
    """
    Decorator for caching API calls.

    Usage:
        @cache_api_call(cache_instance, cost_estimate=0.001)
        def my_api_call(prompt):
            return llm.generate(prompt)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return cache.cached_call(func, *args, cost_per_call=cost_estimate, **kwargs)
        return wrapper
    return decorator


# Global cache instance
_global_cache = None

def get_global_cache() -> APICache:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = APICache()
    return _global_cache


# Demo/Test
if __name__ == "__main__":
    print("API Cache Demo\n")

    cache = APICache(cache_dir=".test_cache", ttl=60)

    # Simulate API calls
    def expensive_api_call(prompt: str) -> str:
        """Simulated API call."""
        print(f"  [API CALL] Processing: {prompt[:50]}...")
        time.sleep(0.1)  # Simulate network delay
        return f"Result for: {prompt}"

    # First call - cache miss
    print("Call 1 (should miss cache):")
    result1 = cache.cached_call(expensive_api_call, "What is Python?")
    print(f"  Result: {result1}\n")

    # Second call - cache hit
    print("Call 2 (should hit cache):")
    result2 = cache.cached_call(expensive_api_call, "What is Python?")
    print(f"  Result: {result2}\n")

    # Different call - cache miss
    print("Call 3 (different input, should miss):")
    result3 = cache.cached_call(expensive_api_call, "What is Java?")
    print(f"  Result: {result3}\n")

    # Show statistics
    stats = cache.get_stats()
    print("Cache Statistics:")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit Rate: {stats['hit_rate']:.1%}")
    print(f"  Saved Calls: {stats['saved_calls']}")
    print(f"  Estimated Savings: ${stats['estimated_savings_usd']:.4f}")

    # Cleanup
    cache.clear()
    import shutil
    shutil.rmtree(".test_cache")

    print("\nDemo complete!")
