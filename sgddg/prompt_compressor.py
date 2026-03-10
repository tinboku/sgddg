"""
Prompt compressor for SGDDG.

Reduces token consumption by filtering redundant metadata, compressing
statistics, and removing low-confidence KG injections before prompt assembly.
"""

import json
from enum import Enum
from typing import Dict, Any, List, Optional


class CompressionLevel(Enum):
    NONE = 0       # No compression
    LIGHT = 1      # Remove redundant stats
    BALANCED = 2   # Compress KG + stats
    AGGRESSIVE = 3 # Keep only core columns and high-confidence KG


class PromptCompressor:
    """Compresses column profiles and KG matches to reduce prompt token count."""

    def __init__(self, level: CompressionLevel = CompressionLevel.BALANCED):
        self.level = level

    def compress_column_profile(
        self,
        profile: Dict[str, Any],
        level: Optional[CompressionLevel] = None
    ) -> Dict[str, Any]:
        """Compress a single column profile based on compression level."""
        target_level = level or self.level
        if target_level == CompressionLevel.NONE:
            return profile

        compressed = {
            "column_name": profile.get("column_name"),
            "data_type": profile.get("data_type"),
        }

        # Always keep basic stats
        basics = ["null_count", "null_rate", "unique_count", "unique_ratio"]
        for key in basics:
            if key in profile:
                compressed[key] = profile[key]

        # Numeric columns
        if profile.get("is_numeric"):
            stats = ["min", "max", "mean"]
            if target_level.value < CompressionLevel.BALANCED.value:
                stats.extend(["std", "median"])

            for key in stats:
                if key in profile:
                    val = profile[key]
                    if isinstance(val, float):
                        compressed[key] = round(val, 2)
                    else:
                        compressed[key] = val

        # Sample values compression
        samples = profile.get("sample_values", [])
        if target_level.value >= CompressionLevel.AGGRESSIVE.value:
            compressed["sample_values"] = samples[:2]
        elif target_level.value >= CompressionLevel.BALANCED.value:
            compressed["sample_values"] = samples[:5]
        else:
            compressed["sample_values"] = samples[:10]

        return compressed

    def compress_kg_matches(
        self,
        kg_match: Dict[str, Any],
        level: Optional[CompressionLevel] = None
    ) -> Optional[Dict[str, Any]]:
        """Compress KG match results, respecting routing and tier decisions."""
        target_level = level or self.level

        # Skip non-matched entries
        status = kg_match.get("status", "no_match")
        if status in ("no_match", "error", "skipped", "filtered"):
            return None if target_level.value >= CompressionLevel.BALANCED.value else kg_match

        # Respect should_inject flag from filter/router
        if kg_match.get("should_inject") is False:
            return None

        # Uncertain matches get minimal injection
        if status == "uncertain":
            if target_level.value >= CompressionLevel.BALANCED.value:
                return None
            return {
                "concept": kg_match.get("concept", {}).get("display_name", "unknown"),
                "strength": "UNCERTAIN",
                "score": round(kg_match.get("score", 0.0), 2),
                "note": "low confidence - use with caution"
            }

        # Tiered injection based on semantic strength
        strength = kg_match.get("semantic_strength", "CONTEXTUAL")
        if target_level == CompressionLevel.AGGRESSIVE and strength not in ("EXACT", "RELATED"):
            return None

        concept = kg_match.get("concept", {})
        compressed = {
            "concept": concept.get("display_name"),
            "strength": strength,
            "score": round(kg_match.get("score", 0.0), 2)
        }

        tier = kg_match.get("tier")
        if tier:
            compressed["tier"] = tier

        # Description length depends on strength and compression level
        desc = concept.get("description", "") or concept.get("definition", "")
        if desc:
            if strength == "EXACT":
                max_len = 100 if target_level == CompressionLevel.AGGRESSIVE else 300
                compressed["description"] = desc[:max_len] + "..." if len(desc) > max_len else desc
            elif strength == "RELATED":
                max_len = 80 if target_level == CompressionLevel.AGGRESSIVE else 150
                compressed["description"] = desc[:max_len] + "..." if len(desc) > max_len else desc
            else:
                if target_level.value < CompressionLevel.BALANCED.value:
                    compressed["description"] = desc[:100] + "..." if len(desc) > 100 else desc

        # Include aliases for high-strength matches
        aliases = concept.get("aliases", [])
        if aliases and strength in ("EXACT", "RELATED"):
            compressed["aliases"] = aliases[:5]

        return compressed

    def format_batch_for_prompt(
        self,
        columns_data: List[Dict[str, Any]],
        level: Optional[CompressionLevel] = None
    ) -> str:
        """Compress and format multiple columns as a JSON string for the prompt."""
        target_level = level or self.level
        processed = []

        for item in columns_data:
            col_name = item["column_name"]
            profile = item["profile"]
            kg = item["kg_match"]

            p_comp = self.compress_column_profile(profile, target_level)
            k_comp = self.compress_kg_matches(kg, target_level)

            entry = {
                "column": col_name,
                "stats": p_comp,
            }
            if k_comp:
                entry["knowledge_graph"] = k_comp

            processed.append(entry)

        return json.dumps(processed, indent=2 if target_level.value < CompressionLevel.BALANCED.value else None)


def compress_prompt_tokens(
    original_data: List[Dict],
    level: CompressionLevel = CompressionLevel.BALANCED
) -> str:
    """Convenience function to compress prompt data."""
    compressor = PromptCompressor(level)
    return compressor.format_batch_for_prompt(original_data)


if __name__ == "__main__":
    test_data = [
        {
            "column_name": "Revenue",
            "profile": {
                "column_name": "Revenue",
                "data_type": "float64",
                "is_numeric": True,
                "null_count": 0,
                "unique_count": 1000,
                "min": 0,
                "max": 1000000,
                "mean": 500000,
                "std": 200000,
                "sample_values": [123.45, 678.9, 1011.12, 1314.15, 1617.18, 1920.21]
            },
            "kg_match": {
                "status": "matched",
                "semantic_strength": "EXACT",
                "score": 0.98,
                "concept": {
                    "display_name": "Revenue",
                    "description": "Income that a business has from its normal business activities, usually from the sale of goods and services to customers. It is also called sales or turnover."
                }
            }
        }
    ]

    compressor = PromptCompressor(CompressionLevel.BALANCED)
    print("=== Balanced Compression ===")
    print(compressor.format_batch_for_prompt(test_data))

    compressor.level = CompressionLevel.AGGRESSIVE
    print("\n=== Aggressive Compression ===")
    print(compressor.format_batch_for_prompt(test_data))
