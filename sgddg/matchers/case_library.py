"""
Case library for Case-Based Semantic Fingerprinting (CBSF).

Stores and retrieves historical fingerprint cases, enabling similarity-based
matching of new columns against known semantic labels.
"""

import json
import os
import numpy as np
from typing import List, Dict, Any, Optional

class CaseLibrary:
    """Manages a library of column fingerprint cases for similarity retrieval."""

    def __init__(self, storage_path: str = "case_library.json"):
        self.storage_path = storage_path
        self.cases: List[Dict[str, Any]] = []
        self.load()

    def add_case(self, fingerprint: Dict[str, Any], semantic_label: str):
        """Add a labeled fingerprint case to the library.

        Args:
            fingerprint: Fingerprint dict from ColumnFingerprint.extract()
            semantic_label: Ground-truth semantic label (e.g. 'Price', 'PassengerAge')
        """
        case = {
            "fingerprint": fingerprint,
            "label": semantic_label
        }
        self.cases.append(case)
        self.save()

    def find_nearest_cases(self, query_fingerprint: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
        """Find the most similar historical cases by weighted feature distance.

        Uses unique_ratio (weight 0.6) and skewness (weight 0.4) with a hard
        filter on data type match.
        """
        results = []

        q_stats = query_fingerprint["stats"]
        q_type = query_fingerprint["data_type"]

        for case in self.cases:
            c_fp = case["fingerprint"]
            c_stats = c_fp["stats"]

            # Hard filter: types must match
            if c_fp["data_type"] != q_type:
                continue

            # Weighted distance (lower = more similar)
            dist = abs(c_stats["unique_ratio"] - q_stats["unique_ratio"]) * 0.6 + \
                   abs(c_stats["skewness"] - q_stats["skewness"]) * 0.4

            results.append({
                "label": case["label"],
                "distance": round(dist, 4),
                "case_name": c_fp["col_name"]
            })

        results.sort(key=lambda x: x["distance"])
        return results[:top_k]

    def save(self):
        """Persist the case library to disk."""
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.cases, f, indent=2, ensure_ascii=False)

    def load(self):
        """Load the case library from disk."""
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r", encoding="utf-8") as f:
                self.cases = json.load(f)

if __name__ == "__main__":
    lib = CaseLibrary("seed_cases.json")

    # Add a seed case
    lib.add_case({
        "col_name": "historical_fare",
        "data_type": "float64",
        "stats": {"unique_ratio": 0.27, "skewness": 4.7},
        "patterns": {}
    }, "Transportation Fare (Money)")

    # Query with a similar fingerprint
    query = {
        "col_name": "current_fare",
        "data_type": "float64",
        "stats": {"unique_ratio": 0.28, "skewness": 4.5},
        "patterns": {}
    }

    print("\nSearching for similar historical cases by fingerprint...")
    matches = lib.find_nearest_cases(query)
    for m in matches:
        print(f"Match: {m['label']} | Distance: {m['distance']} (from: {m['case_name']})")
