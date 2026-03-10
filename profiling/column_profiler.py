"""Column Profiler - extracts deep statistical features from dataset columns."""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional
from datetime import datetime


class ColumnProfiler:
    """Column profiler with distribution, temporal, pattern, and constraint detection."""

    # Regex patterns for value format detection
    VALUE_PATTERNS = {
        "email": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        "url": r'^https?://[^\s]+$',
        "phone": r'^[\+]?[\d\s\-\(\)]{7,15}$',
        "ip_address": r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
        "uuid": r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        "currency_code": r'^[A-Z]{3}$',
        "zip_code": r'^\d{5}(-\d{4})?$',
        "hex_color": r'^#[0-9a-fA-F]{6}$',
    }

    def __init__(self):
        self.unit_patterns = {
            "currency": [r"\$", r"USD", r"EUR", r"GBP", r"CNY", r"¥"],
            "percentage": [r"%", r"percent"],
            "time": [r"year", r"month", r"day", r"hour", r"minute", r"second"],
            "distance": [r"km", r"mile", r"meter", r"foot"],
            "weight": [r"kg", r"pound", r"gram"],
        }

    def profile(
        self,
        column: pd.Series,
        column_name: str,
        sample_size: int = 10
    ) -> Dict[str, Any]:
        """
        Generate an enhanced profile for a single column.

        Args:
            column: Pandas Series
            column_name: Column name
            sample_size: Number of sample values

        Returns:
            Enhanced column profile dictionary
        """
        total_count = len(column)
        unique_count = int(column.nunique())

        profile = {
            "column_name": column_name,
            "data_type": self._infer_data_type(column),
            "structural_type": self._infer_structural_type(column),
            "total_count": total_count,
            "null_count": int(column.isnull().sum()),
            "null_rate": float(column.isnull().sum() / total_count) if total_count > 0 else 0.0,
            "unique_count": unique_count,
            "cardinality_ratio": float(unique_count / total_count) if total_count > 0 else 0.0,
            "sample_values": self._get_sample_values(column, sample_size),
            "unit": self._extract_unit(column_name, column),
            "statistics": self._compute_statistics(column),
            "inferred_semantic_type": self._infer_semantic_type(column_name, column),
            # Enhanced features (Datamart-style)
            "distribution_type": self._detect_distribution_type(column),
            "temporal_resolution": self._detect_temporal_resolution(column_name, column),
            "value_pattern": self._detect_value_patterns(column),
            "value_range_semantics": self._detect_value_range_semantics(column),
            # Schema constraints (for Schema-Guided approach)
            "constraints": self._detect_constraints(column, column_name),
        }

        return profile

    def _infer_structural_type(self, column: pd.Series) -> str:
        """
        Infer structural type (e.g., Integer, Float, Text, Date).
        Reference: AutoDDG/datamart_profiler structural_type
        """
        if pd.api.types.is_integer_dtype(column):
            return "Integer"
        elif pd.api.types.is_float_dtype(column):
            return "Float"
        elif pd.api.types.is_bool_dtype(column):
            return "Boolean"
        elif pd.api.types.is_datetime64_any_dtype(column):
            return "DateTime"
        else:
            return "Text"

    def _infer_data_type(self, column: pd.Series) -> str:
        """Infer the data type of a column."""
        dtype = str(column.dtype)

        if "int" in dtype:
            return "integer"
        elif "float" in dtype:
            return "float"
        elif "bool" in dtype:
            return "boolean"
        elif "datetime" in dtype or "date" in dtype:
            return "datetime"
        elif "object" in dtype or "string" in dtype:
            non_null = column.dropna()
            if len(non_null) == 0:
                return "string"
            try:
                pd.to_datetime(non_null.iloc[0])
                return "date_string"
            except Exception:
                pass
            return "string"
        else:
            return "unknown"

    def _get_sample_values(self, column: pd.Series, sample_size: int) -> List[Any]:
        """Get representative sample values from the column."""
        non_null = column.dropna()
        if len(non_null) == 0:
            return []

        sample = non_null.sample(min(sample_size, len(non_null))).tolist()

        serializable_sample = []
        for val in sample:
            if isinstance(val, (np.integer, np.int64)):
                serializable_sample.append(int(val))
            elif isinstance(val, (np.floating, np.float64)):
                serializable_sample.append(float(val))
            elif pd.isna(val):
                continue
            else:
                serializable_sample.append(str(val))

        return serializable_sample

    def _extract_unit(self, column_name: str, column: pd.Series) -> Optional[str]:
        """Extract unit information from column name or values."""
        for unit_type, patterns in self.unit_patterns.items():
            for pattern in patterns:
                if re.search(pattern, column_name, re.IGNORECASE):
                    return unit_type

        if column.dtype == "object":
            sample = column.dropna().head(5).astype(str)
            for val in sample:
                for unit_type, patterns in self.unit_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, val, re.IGNORECASE):
                            return unit_type

        return None

    def _compute_statistics(self, column: pd.Series) -> Dict[str, Any]:
        """Compute statistical summaries for the column."""
        stats = {}

        if pd.api.types.is_numeric_dtype(column):
            non_null = column.dropna()
            if not non_null.empty:
                stats["mean"] = float(non_null.mean())
                stats["std"] = float(non_null.std())
                stats["min"] = float(non_null.min())
                stats["max"] = float(non_null.max())
                stats["median"] = float(non_null.median())
                stats["q25"] = float(non_null.quantile(0.25))
                stats["q75"] = float(non_null.quantile(0.75))
                stats["skewness"] = float(non_null.skew())
                stats["kurtosis"] = float(non_null.kurtosis())

        elif column.dtype == "object" or column.dtype.name == "category":
            value_counts = column.value_counts()
            stats["top_values"] = value_counts.head(5).to_dict()
            stats["cardinality"] = len(value_counts)
            # Average string length for text columns
            non_null = column.dropna().astype(str)
            if not non_null.empty:
                stats["avg_string_length"] = float(non_null.str.len().mean())

        return stats

    def _infer_semantic_type(self, column_name: str, column: pd.Series) -> str:
        """Infer the semantic type based on column name and data patterns."""
        name_lower = column_name.lower()

        if any(kw in name_lower for kw in ["id", "key", "code", "number"]):
            return "identifier"
        if any(kw in name_lower for kw in ["date", "time", "year", "month", "day"]):
            return "temporal"
        if any(kw in name_lower for kw in ["city", "country", "state", "location", "address"]):
            return "spatial"
        if any(kw in name_lower for kw in ["revenue", "profit", "sales", "amount", "price", "cost"]):
            return "metric"
        if any(kw in name_lower for kw in ["rate", "ratio", "percentage", "%", "margin"]):
            return "ratio"
        if any(kw in name_lower for kw in ["count", "number", "quantity"]):
            return "count"
        if column.dtype == "object" and column.nunique() < len(column) * 0.5:
            return "categorical"
        if pd.api.types.is_numeric_dtype(column):
            return "numerical"

        return "general"

    def _detect_distribution_type(self, column: pd.Series) -> Optional[str]:
        """
        Detect the distribution type of a numeric column.
        Uses scipy.stats.normaltest + skewness to classify as
        Gaussian/Uniform/Skewed/Bimodal/Zipf.
        """
        if not pd.api.types.is_numeric_dtype(column):
            return None

        non_null = column.dropna()
        if len(non_null) < 8:
            return "insufficient_data"

        try:
            from scipy import stats as scipy_stats

            skewness = float(non_null.skew())
            kurtosis = float(non_null.kurtosis())

            # Normal test (requires n >= 8)
            _, p_value = scipy_stats.normaltest(non_null)

            if p_value > 0.05:
                return "gaussian"

            # Check for uniform distribution
            if abs(skewness) < 0.5 and kurtosis < -1.0:
                return "uniform"

            # Check for heavy skew (potential Zipf/power-law)
            if abs(skewness) > 2.0:
                return "zipf_like"

            # Check for bimodal (negative kurtosis with moderate skew)
            if kurtosis < -0.5 and abs(skewness) < 1.0:
                return "bimodal"

            # General skewed distribution
            if abs(skewness) > 1.0:
                return "skewed_right" if skewness > 0 else "skewed_left"

            return "non_gaussian"

        except ImportError:
            # scipy not available - use basic heuristics
            skewness = float(non_null.skew())
            if abs(skewness) < 0.5:
                return "approximately_symmetric"
            elif abs(skewness) > 2.0:
                return "heavily_skewed"
            else:
                return "moderately_skewed"
        except Exception:
            return None

    def _detect_temporal_resolution(
        self, column_name: str, column: pd.Series
    ) -> Optional[str]:
        """
        Detect temporal resolution for date/time columns.
        Returns: Year/Quarter/Month/Week/Day/Hour/Minute/Timestamp or None.
        """
        name_lower = column_name.lower()

        # Check column name hints
        if "year" in name_lower:
            return "year"
        if "quarter" in name_lower:
            return "quarter"
        if "month" in name_lower:
            return "month"
        if "week" in name_lower:
            return "week"
        if "day" in name_lower:
            return "day"
        if "hour" in name_lower:
            return "hour"
        if "minute" in name_lower:
            return "minute"

        # If it's a datetime column or parseable date string, detect resolution from data
        if pd.api.types.is_datetime64_any_dtype(column):
            return self._resolution_from_datetime(column)

        # Try parsing string columns as dates
        if column.dtype == "object":
            non_null = column.dropna().head(20)
            if len(non_null) == 0:
                return None
            try:
                parsed = pd.to_datetime(non_null, infer_datetime_format=True)
                return self._resolution_from_datetime(parsed)
            except Exception:
                pass

        # Integer columns that look like years (e.g., 1990-2030)
        if pd.api.types.is_integer_dtype(column):
            non_null = column.dropna()
            if not non_null.empty:
                min_val = int(non_null.min())
                max_val = int(non_null.max())
                if 1900 <= min_val <= 2100 and 1900 <= max_val <= 2100:
                    return "year"

        return None

    def _resolution_from_datetime(self, dt_series: pd.Series) -> str:
        """Determine temporal resolution from a datetime Series."""
        non_null = dt_series.dropna()
        if len(non_null) < 2:
            return "unknown"

        # Check if all times are midnight (date-only)
        has_time = any(
            t.hour != 0 or t.minute != 0 or t.second != 0
            for t in non_null.head(10)
        )

        if not has_time:
            # Check if all days are 1st (month-level)
            if all(t.day == 1 for t in non_null.head(10)):
                if all(t.month == 1 for t in non_null.head(10)):
                    return "year"
                return "month"
            return "day"
        else:
            # Has time component
            has_seconds = any(t.second != 0 for t in non_null.head(10))
            has_minutes = any(t.minute != 0 for t in non_null.head(10))
            if has_seconds:
                return "timestamp"
            elif has_minutes:
                return "minute"
            else:
                return "hour"

    def _detect_value_patterns(self, column: pd.Series) -> Optional[str]:
        """
        Detect common value patterns in string columns using regex.
        Returns the detected pattern type (email, url, phone, ip, uuid, etc.) or None.
        """
        if column.dtype != "object":
            return None

        non_null = column.dropna().astype(str)
        if len(non_null) == 0:
            return None

        # Sample up to 20 values for pattern matching
        sample = non_null.head(20)

        for pattern_name, pattern_regex in self.VALUE_PATTERNS.items():
            match_count = sum(
                1 for val in sample if re.match(pattern_regex, val.strip(), re.IGNORECASE)
            )
            # If >50% of samples match, declare the pattern
            if match_count > len(sample) * 0.5:
                return pattern_name

        return None

    def _detect_value_range_semantics(self, column: pd.Series) -> Optional[str]:
        """
        Infer semantic meaning from the numeric value range.
        Examples:
        - 0-100 → possibly percentage
        - 0-1 → possibly ratio/probability
        - Contains negatives → possibly profit/loss or delta
        - All positive integers → possibly count
        """
        if not pd.api.types.is_numeric_dtype(column):
            return None

        non_null = column.dropna()
        if non_null.empty:
            return None

        min_val = float(non_null.min())
        max_val = float(non_null.max())
        has_negatives = min_val < 0

        # 0 to 1 range → ratio or probability
        if 0.0 <= min_val and max_val <= 1.0:
            return "ratio_or_probability"

        # 0 to 100 range → likely percentage
        if 0.0 <= min_val and max_val <= 100.0:
            return "possible_percentage"

        # Contains negatives → profit/loss, delta, or temperature
        if has_negatives:
            return "signed_value_profit_loss_or_delta"

        # All positive integers with high values → monetary or count
        if min_val >= 0 and pd.api.types.is_integer_dtype(column):
            if max_val > 1000000:
                return "large_positive_possibly_monetary"
            elif max_val > 100:
                return "positive_integer_possibly_count"

        # Large float values → monetary
        if min_val >= 0 and max_val > 10000:
            return "large_positive_possibly_monetary"

        return None

    def _detect_constraints(self, column: pd.Series, column_name: str) -> Dict[str, Any]:
        """
        Detect schema constraints on this column.
        Returns information about NOT NULL, UNIQUE, PRIMARY KEY, FOREIGN KEY, and CHECK constraints.
        """
        constraints = {}

        # 1. Nullable constraint
        has_nulls = column.isnull().any()
        constraints['nullable'] = bool(has_nulls)
        constraints['not_null'] = not has_nulls

        # 2. Uniqueness constraint
        total_count = len(column)
        unique_count = column.nunique()
        is_unique = (unique_count == total_count) and total_count > 0
        constraints['unique'] = bool(is_unique)

        # 3. Primary key candidate detection
        # A PK should be: unique + non-null + has 'id'/'key' in name
        is_pk_candidate = (
            is_unique and
            not has_nulls and
            self._is_pk_candidate_by_name(column_name) and
            total_count > 0
        )
        constraints['primary_key_candidate'] = bool(is_pk_candidate)

        # 4. Foreign key candidate detection
        # An FK should: have 'id'/'key' in name + not necessarily unique + appropriate data type
        is_fk_candidate = (
            self._is_fk_candidate_by_name(column_name) and
            not is_pk_candidate  # Exclude PKs
        )
        constraints['foreign_key_candidate'] = bool(is_fk_candidate)

        # 5. Check constraints (range-based)
        check_constraints = self._infer_range_constraints(column)
        if check_constraints:
            constraints['check_constraints'] = check_constraints

        # 6. Default value inference (most common value if very dominant)
        if total_count > 0:
            non_null = column.dropna()
            if len(non_null) > 0:
                value_counts = non_null.value_counts()
                if len(value_counts) > 0:
                    most_common_value = value_counts.index[0]
                    most_common_ratio = value_counts.iloc[0] / len(non_null)
                    # If > 80% of values are the same, it might be a default
                    if most_common_ratio > 0.8:
                        constraints['potential_default'] = {
                            'value': str(most_common_value),
                            'frequency': float(most_common_ratio)
                        }

        return constraints

    def _is_pk_candidate_by_name(self, column_name: str) -> bool:
        """Check if column name suggests it's a primary key."""
        name_lower = column_name.lower()
        # Common PK patterns
        pk_patterns = [
            r'^id$',
            r'^.*_id$',
            r'^pk_.*$',
            r'^primary_key$',
            r'^.*_key$',
        ]
        return any(re.match(pattern, name_lower) for pattern in pk_patterns)

    def _is_fk_candidate_by_name(self, column_name: str) -> bool:
        """Check if column name suggests it's a foreign key."""
        name_lower = column_name.lower()
        # Common FK patterns (excluding bare 'id')
        fk_patterns = [
            r'^.*_id$',
            r'^.*_key$',
            r'^fk_.*$',
            r'^.*_code$',
            r'^.*_ref$',
        ]
        # Exclude bare 'id' (likely a PK)
        if name_lower == 'id':
            return False
        return any(re.match(pattern, name_lower) for pattern in fk_patterns)

    def _infer_range_constraints(self, column: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Infer CHECK constraints based on the data range.
        For numeric columns, detect if values are always positive, within a range, etc.
        """
        if not pd.api.types.is_numeric_dtype(column):
            return None

        non_null = column.dropna()
        if non_null.empty:
            return None

        min_val = float(non_null.min())
        max_val = float(non_null.max())

        constraints = {}

        # Check for positivity constraint
        if min_val >= 0:
            constraints['non_negative'] = True
            if min_val > 0:
                constraints['positive'] = True

        # Check for specific ranges
        if 0 <= min_val and max_val <= 1:
            constraints['range'] = 'between_0_and_1'
            constraints['description'] = 'Value must be between 0 and 1 (ratio/probability)'
        elif 0 <= min_val and max_val <= 100:
            constraints['range'] = 'between_0_and_100'
            constraints['description'] = 'Value must be between 0 and 100 (possibly percentage)'
        elif min_val >= 0 and max_val <= 1000000:
            constraints['range'] = 'positive_bounded'
            constraints['description'] = f'Value must be between 0 and {int(max_val)}'

        # Integer-specific constraints
        if pd.api.types.is_integer_dtype(column):
            # Check if it's a boolean disguised as int (0/1 only)
            unique_vals = set(non_null.unique())
            if unique_vals.issubset({0, 1}):
                constraints['boolean_encoded'] = True
                constraints['description'] = 'Binary value (0 or 1)'

        return constraints if constraints else None


# Minimal Demo
if __name__ == "__main__":
    print("ColumnProfiler Enhanced Demo")

    df = pd.DataFrame({
        "Year": [2020, 2021, 2022, 2023],
        "Revenue ($M)": [100.5, 120.3, 135.7, 150.2],
        "Operating Margin (%)": [15.2, 16.8, 17.5, 18.1],
        "Country": ["USA", "USA", "China", "China"],
        "ID": ["A001", "A002", "A003", "A004"]
    })

    profiler = ColumnProfiler()

    print("\nEnhanced Column Profiling:")
    for column_name in df.columns:
        profile = profiler.profile(df[column_name], column_name)

        print(f"\nColumn: {profile['column_name']}")
        print(f"  Data Type: {profile['data_type']}")
        print(f"  Semantic Type: {profile['inferred_semantic_type']}")
        print(f"  Unit: {profile['unit']}")
        print(f"  Cardinality Ratio: {profile['cardinality_ratio']:.2f}")
        print(f"  Distribution: {profile['distribution_type']}")
        print(f"  Temporal Resolution: {profile['temporal_resolution']}")
        print(f"  Value Pattern: {profile['value_pattern']}")
        print(f"  Value Range Semantics: {profile['value_range_semantics']}")

    print("\nDemo complete!")
