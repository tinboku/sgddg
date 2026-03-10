"""
Statistical fingerprint extractor for Case-Based Semantic Fingerprinting (CBSF).

Converts a data column into a feature vector capturing statistical properties
(uniqueness, skewness, kurtosis) and string patterns.
"""

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import re
from typing import Dict, Any

class ColumnFingerprint:
    """Extracts statistical fingerprints from data columns."""

    @staticmethod
    def extract(series: pd.Series, col_name: str) -> Dict[str, Any]:
        """Extract a fingerprint dict with stats and pattern features."""

        # Basic physical features
        total_count = len(series)
        valid_series = series.dropna()
        valid_count = len(valid_series)
        null_rate = (total_count - valid_count) / total_count if total_count > 0 else 0
        unique_count = valid_series.nunique()
        unique_ratio = unique_count / valid_count if valid_count > 0 else 0

        # Statistical features (numeric only)
        mean, std, skewness, kurt = 0, 0, 0, 0
        is_numeric = pd.api.types.is_numeric_dtype(series)

        if is_numeric and valid_count > 0:
            mean = valid_series.mean()
            std = valid_series.std()
            skewness = skew(valid_series) if valid_count > 2 else 0
            kurt = kurtosis(valid_series) if valid_count > 2 else 0

        # Pattern features (string only)
        avg_len, is_id_like = 0, 0
        if not is_numeric and valid_count > 0:
            avg_len = valid_series.astype(str).str.len().mean()
            # Simple heuristic: mixed alphanumeric with fixed length suggests an ID
            sample_val = str(valid_series.iloc[0])
            if re.search(r'[0-9]', sample_val) and re.search(r'[a-zA-Z]', sample_val):
                is_id_like = 1

        return {
            "col_name": col_name,
            "data_type": str(series.dtype),
            "stats": {
                "unique_ratio": round(unique_ratio, 4),
                "null_rate": round(null_rate, 4),
                "mean": round(float(mean), 2),
                "std": round(float(std), 2),
                "skewness": round(float(skewness), 2),
                "kurtosis": round(float(kurt), 2)
            },
            "patterns": {
                "avg_length": round(avg_len, 2),
                "is_id_like": is_id_like
            }
        }

if __name__ == "__main__":
    df = pd.DataFrame({
        "Age": [22, 38, 26, 35, np.nan],
        "Fare": [7.25, 71.28, 7.92, 53.10, 8.05],
        "Ticket": ["A/5 21171", "PC 17599", "STON/O2. 3101282", "113803", "373450"]
    })

    for col in df.columns:
        print(f"\n{col} Fingerprint:")
        fingerprint = ColumnFingerprint.extract(df[col], col)
        print(fingerprint)


Column relation detector for discovering inter-column relationships.

Detects correlations and functional dependencies between columns,
serving as the foundation for schema graph construction.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

class ColumnRelationDetector:
    """Detects structural relationships between columns in a dataset."""

    @staticmethod
    def detect_relations(df: pd.DataFrame) -> List[Dict]:
        """Scan all column pairs and detect correlations and functional dependencies.

        Returns:
            List of relation dicts: {'source', 'target', 'type', 'weight'}
        """
        relations = []
        columns = df.columns

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col_a = columns[i]
                col_b = columns[j]

                # 1. Numeric correlation (Pearson)
                if pd.api.types.is_numeric_dtype(df[col_a]) and pd.api.types.is_numeric_dtype(df[col_b]):
                    valid_df = df[[col_a, col_b]].dropna()
                    if len(valid_df) > 5:
                        corr = valid_df[col_a].corr(valid_df[col_b])
                        if abs(corr) > 0.5:
                            relations.append({
                                "source": col_a,
                                "target": col_b,
                                "type": "correlation",
                                "weight": round(corr, 3)
                            })

                # 2. Functional dependency
                # If each value of A maps to nearly one unique value of B,
                # then B is functionally dependent on A.
                if df[col_a].nunique() > 0:
                    unique_mappings = df.groupby(col_a)[col_b].nunique().mean()
                    if unique_mappings < 1.05 and df[col_a].nunique() > 2:
                        relations.append({
                            "source": col_a,
                            "target": col_b,
                            "type": "functional_dependency",
                            "weight": 1.0
                        })

        return relations

if __name__ == "__main__":
    data = {
        "PassengerId": [1, 2, 3, 4, 5],
        "Survived": [0, 1, 1, 0, 1],
        "Pclass": [3, 1, 3, 1, 3],
        "Fare": [7.25, 71.28, 7.92, 53.10, 8.05],
        "Age": [22, 38, 26, 35, 27]
    }
    test_df = pd.DataFrame(data)

    print("Detecting inter-column relationships...")
    detector = ColumnRelationDetector()
    results = detector.detect_relations(test_df)

    for rel in results:
        print(f"Found: {rel['source']} --[{rel['type']}]--> {rel['target']} (Strength: {rel['weight']})")
