"""
Caching system for SGDDG.

Provides disk-backed caches for physical profiles, KG match results,
and full metadata to avoid redundant computation and API calls.
"""

import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd


class ProfileCache:
    """Disk-backed cache for column physical profiles, keyed by data fingerprint."""

    def __init__(
        self,
        cache_dir: str = "./cache/profiles",
        ttl_hours: int = 168  # 7 days
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600 if ttl_hours > 0 else None

        self.hits = 0
        self.misses = 0

        print(f"[Cache] Profile Cache initialized")
        print(f"   - Directory: {self.cache_dir}")
        print(f"   - TTL: {ttl_hours} hours" if ttl_hours > 0 else "   - TTL: Never expire")

    def _compute_fingerprint(self, df_column: pd.Series) -> str:
        """Compute an MD5 fingerprint from column name, dtype, and first 100 non-null values."""
        col_name = str(df_column.name)
        data_type = str(df_column.dtype)

        sample_values = df_column.dropna().head(100).tolist()
        sample_json = json.dumps(sample_values, sort_keys=True, default=str)

        fingerprint_str = f"{col_name}|{data_type}|{sample_json}"
        return hashlib.md5(fingerprint_str.encode('utf-8')).hexdigest()

    def get(self, df_column: pd.Series) -> Optional[Dict[str, Any]]:
        """Retrieve a cached profile, or None on miss/expiry."""
        if self is None: return None

        fingerprint = self._compute_fingerprint(df_column)
        cache_file = self.cache_dir / f"{fingerprint}.pkl"

        if not cache_file.exists():
            self.misses += 1
            return None

        if self.ttl_seconds is not None:
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age > self.ttl_seconds:
                cache_file.unlink()
                self.misses += 1
                return None

        try:
            with open(cache_file, 'rb') as f:
                profile = pickle.load(f)
            self.hits += 1
            return profile
        except Exception as e:
            print(f"    Warning: Failed to load cache for {df_column.name}: {e}")
            self.misses += 1
            return None

    def save(self, df_column: pd.Series, profile: Dict[str, Any]):
        """Persist a profile to the cache."""
        if self is None: return

        fingerprint = self._compute_fingerprint(df_column)
        cache_file = self.cache_dir / f"{fingerprint}.pkl"

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(profile, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"    Warning: Failed to save cache for {df_column.name}: {e}")

    def clear(self):
        """Remove all cached profile files."""
        count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
            count += 1

        self.hits = 0
        self.misses = 0

        print(f"Cleared: Cleared {count} cache files")

    def get_stats(self) -> Dict[str, Any]:
        """Return cache hit/miss statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        file_count = len(list(self.cache_dir.glob("*.pkl")))

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "total_files": file_count
        }

    def print_stats(self):
        """Print cache statistics to stdout."""
        stats = self.get_stats()
        print(f"\n[Stats] Profile Cache Statistics:")
        print(f"   - Hits: {stats['hits']}")
        print(f"   - Misses: {stats['misses']}")
        print(f"   - Hit Rate: {stats['hit_rate']:.1%}")
        print(f"   - Total Cached Files: {stats['total_files']}")


class KGMatchCache:
    """Disk-backed cache for KG matching results, keyed by column name and sample values."""

    def __init__(
        self,
        cache_dir: str = "./cache/kg_matches",
        ttl_hours: int = 168
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600 if ttl_hours > 0 else None

        self.hits = 0
        self.misses = 0

        print(f"[Cache] KG Match Cache initialized")
        print(f"   - Directory: {self.cache_dir}")

    def _compute_key(self, column_name: str, sample_values: list) -> str:
        """Hash column name + first 10 sample values into a cache key."""
        samples = sample_values[:10] if sample_values else []
        key_str = f"{column_name}|{json.dumps(samples, sort_keys=True, default=str)}"
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()

    def get(
        self,
        column_name: str,
        sample_values: list
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a cached KG match result, or None on miss/expiry."""
        if self is None: return None

        key = self._compute_key(column_name, sample_values)
        cache_file = self.cache_dir / f"{key}.json"

        if not cache_file.exists():
            self.misses += 1
            return None

        if self.ttl_seconds is not None:
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age > self.ttl_seconds:
                cache_file.unlink()
                self.misses += 1
                return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
            self.hits += 1
            return result
        except Exception as e:
            print(f"    Warning: Failed to load KG match cache: {e}")
            self.misses += 1
            return None

    def save(
        self,
        column_name: str,
        sample_values: list,
        match_result: Dict[str, Any]
    ):
        """Persist a KG match result to the cache."""
        if self is None: return

        key = self._compute_key(column_name, sample_values)
        cache_file = self.cache_dir / f"{key}.json"

        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(match_result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"    Warning: Failed to save KG match cache: {e}")

    def clear(self):
        """Remove all cached KG match files."""
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1

        self.hits = 0
        self.misses = 0

        print(f"Cleared: Cleared {count} KG match cache files")

    def get_stats(self) -> Dict[str, Any]:
        """Return cache hit/miss statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "total_files": len(list(self.cache_dir.glob("*.json")))
        }

    def print_stats(self):
        """Print cache statistics to stdout."""
        stats = self.get_stats()
        print(f"\n[Stats] KG Match Cache Statistics:")
        print(f"   - Hits: {stats['hits']}")
        print(f"   - Misses: {stats['misses']}")
        print(f"   - Hit Rate: {stats['hit_rate']:.1%}")
        print(f"   - Total Cached Files: {stats['total_files']}")


class MetadataCache:
    """Disk-backed cache for complete dataset metadata (UFD/SFD)."""

    def __init__(
        self,
        cache_dir: str = "./cache/metadata",
        ttl_hours: int = 24
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600 if ttl_hours > 0 else None

        print(f"[Cache] Metadata Cache initialized")
        print(f"   - Directory: {self.cache_dir}")

    def _compute_dataset_fingerprint(
        self,
        dataset_name: str,
        column_names: list
    ) -> str:
        """Hash dataset name + sorted column names."""
        fingerprint_str = f"{dataset_name}|{json.dumps(sorted(column_names))}"
        return hashlib.md5(fingerprint_str.encode('utf-8')).hexdigest()

    def get(
        self,
        dataset_name: str,
        column_names: list
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached metadata for a dataset, or None on miss/expiry."""
        fingerprint = self._compute_dataset_fingerprint(dataset_name, column_names)
        cache_file = self.cache_dir / f"{fingerprint}.json"

        if not cache_file.exists():
            return None

        if self.ttl_seconds is not None:
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age > self.ttl_seconds:
                cache_file.unlink()
                return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"    Warning: Failed to load metadata cache: {e}")
            return None

    def save(
        self,
        dataset_name: str,
        column_names: list,
        metadata: Dict[str, Any]
    ):
        """Persist complete metadata to the cache."""
        fingerprint = self._compute_dataset_fingerprint(dataset_name, column_names)
        cache_file = self.cache_dir / f"{fingerprint}.json"

        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"    Warning: Failed to save metadata cache: {e}")

    def clear(self):
        """Remove all cached metadata files."""
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1
        print(f"Cleared: Cleared {count} metadata cache files")


def clear_all_caches():
    """Clear all SGDDG caches."""
    print("Cleared: Clearing all SGDDG caches...")

    ProfileCache().clear()
    KGMatchCache().clear()
    MetadataCache().clear()

    print("Done: All caches cleared")


def print_all_cache_stats():
    """Print statistics for all caches."""
    profile_cache = ProfileCache()
    kg_cache = KGMatchCache()

    print("\n" + "=" * 50)
    print("[Stats] SGDDG Cache Statistics")
    print("=" * 50)

    profile_cache.print_stats()
    kg_cache.print_stats()

    print("=" * 50)


if __name__ == "__main__":
    print("SGDDG Cache System Demo\n")

    import pandas as pd

    df = pd.DataFrame({
        "Revenue": [1000, 1500, 2000],
        "Country": ["USA", "China", "Germany"]
    })

    print("Testing Profile Cache...")
    cache = ProfileCache()

    profile1 = cache.get(df['Revenue'])
    print(f"First attempt: {'HIT' if profile1 else 'MISS'}")

    if profile1 is None:
        cache.save(df['Revenue'], {"column_name": "Revenue", "mean": 1500})

    profile2 = cache.get(df['Revenue'])
    print(f"Second attempt: {'HIT' if profile2 else 'MISS'}")

    cache.print_stats()
    cache.clear()

    print("\nDone: Demo complete")
